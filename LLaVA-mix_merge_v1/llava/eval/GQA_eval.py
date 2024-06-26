from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path

import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mobilevlm.constants import (
    IMAGE_TOKEN_INDEX, 
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, 
     DEFAULT_IM_END_TOKEN,
)
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria

class GQADataset(Dataset):
    def __init__(self, data_path, image_processor, tokenizer, model_config, conv_mode):
        self.image_data = load_dataset("parquet", data_files=str(Path(data_path) / "testdev-00000-of-00001.parquet"))
        self.instr_data = load_dataset("parquet", data_files=str(Path(data_path) / "testdev-00000-of-00001-instruction.parquet"))
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.conv_mode = conv_mode
        self.pair_dict = {}
        for image in self.image_data["train"]:
            self.pair_dict[image["id"]] = image["image"]

    def __len__(self):
        return len(self.instr_data["train"])

    def __getitem__(self, id):
        instr = self.instr_data["train"][id]["question"]
        image = self.pair_dict[self.instr_data["train"][id]["imageId"]]
        answer = self.instr_data["train"][id]["answer"]
        full_answer = self.instr_data["train"][id]["fullAnswer"]
        quest_id = self.instr_data["train"][id]["id"]
        image_id = self.instr_data["train"][id]["imageId"]
        if self.model_config.mm_use_im_start_end:
            instr = DEFAULT_IM_START_TOKEN + \
                    DEFAULT_IMAGE_TOKEN + \
                    DEFAULT_IM_END_TOKEN + \
                    "\n" + instr
        instr = DEFAULT_IMAGE_TOKEN + '\n' + instr + "Answer that question with a lowercase word or a phrase"
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], instr)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # handle the Gray image
        from PIL import Image
        if image.mode == "L":
            image.convert("RGB")
            rgb_image = Image.new("RGB", image.size)
            rgb_image.paste(image)
            image = rgb_image
        image_tensor = process_images([image], self.image_processor, self.model_config)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        
        one_batch = {}
        one_batch["input_ids"] = input_ids
        one_batch["image_tensor"] = image_tensor
        one_batch["answer"] = answer
        one_batch["full_answer"] = full_answer
        one_batch["id"] = quest_id
        one_batch["image_id"] = image_id
        one_batch["image"] = image
        one_batch["question"] = instr
        return one_batch
    

if __name__ == "__main__":
    # receive args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/Work/sparsegpt/SparseGPT-25-MobileVLM_V2-1.7B")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--data-path", type=str, default="/root/autodl-tmp/Work/GQA")
    args = parser.parse_args()

    # load model
    from mobilevlm.model.mobilevlm import load_pretrained_model
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path)
    vision_tower = model.get_vision_tower()
    # if not vision_tower.is_loaded:
    vision_tower.load_model()
    vision_tower.to(device="cuda", dtype=torch.float16)
    # create GQA dataset
    GQA = GQADataset(args.data_path, image_processor, tokenizer, model.config, args.conv_mode)
    GQALoader = DataLoader(GQA, batch_size=1, num_workers=4, shuffle=False)

    # load data and record
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    scores = []
    # cal scores
    def cal_score(pred, ans):
        return float(pred==ans)

    # open js file
    import json
    out_path = "/root/autodl-tmp/Work/MobileVLM/mobilevlm/eval/MobileVLM_V2-1.7B_GQAacc--ttest.json"
    f = open(out_path, "a+", encoding="utf-8")
    for item in tqdm(GQA, total=len(GQA)):
        answer = item["answer"]
        full_answer = item["full_answer"]
        id = item["id"]
        input_ids = item["input_ids"]
        image_tensor = item["image_tensor"]
        question = item["question"]
        # model generate
        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature = args.temperature,
                top_p=args.top_p,
                max_new_tokens=128,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
        input_token_len =input_ids.shape[1]
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip().lower()
        score = cal_score(outputs, answer)
        scores.append(score)
        f.write(json.dumps({
            "question_id": id,
            "score": score,
            "prompt": question,
            "full_answer": full_answer,
            "answer": answer,
            "model_answer": outputs,
            })+"\n")
    print("Acc.:", sum(scores)/len(scores))
    f.write("Acc.: " + str(sum(scores)/len(scores)))
    f.close()
