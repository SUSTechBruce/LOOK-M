from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from PIL import Image
import sys
import os
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mobilevlm.constants import (
    IMAGE_TOKEN_INDEX, 
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, 
     DEFAULT_IM_END_TOKEN,
)
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria

class MMEDataset(Dataset):
    def __init__(self, data_path, image_processor, tokenizer, model_config, conv_mode):
        assert isinstance(data_path, str)
        self.data_list = []
        for file_name in os.listdir(data_path):
            self.data_list.append(load_dataset("parquet", data_files=str(Path(data_path) / file_name)))
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.conv_mode = conv_mode
        self.data_chunck = []
        self.instr_data = []
        self.answer = []
        self.image_t_pair = {}
        for chunck in self.data_list:
            for item in tqdm(chunck["train"], total=len(chunck["train"])):
                self.data_chunck.append({
                    "instr": item["question"],
                    "answer": item["answer"],
                    "image_id": item["question_id"],
                    "category": item["category"],
                })
                if self.data_chunck[-1]["image_id"] not in self.image_t_pair.keys():
                    # handle the Gray image
                    image = item["image"]
                    if image.mode == "L":
                        image.convert("RGB")
                        rgb_image = Image.new("RGB", image.size)
                        rgb_image.paste(image)
                        image = rgb_image
                    image_tensor = process_images([image], self.image_processor, self.model_config)
                    self.image_t_pair[self.data_chunck[-1]["image_id"]] = image_tensor
        pass    
    def __len__(self):
        return len(self.data_chunck)

    def __getitem__(self, id):
        instr = self.data_chunck[id]["instr"]
        image_id = self.data_chunck[id]["image_id"]
        image_tensor = self.image_t_pair[image_id]
        answer = self.data_chunck[id]["answer"]
        category = self.data_chunck[id]["category"]

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
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        
        batch = {}
        batch["input_ids"] = input_ids
        batch["image_tensor"] = image_tensor
        batch["answer"] = answer.lower()
        batch["image_id"] = image_id
        batch["question"] = instr
        batch["category"] = category
        return batch
    

if __name__ == "__main__":
    # receive args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/Work/sparsegpt/SparseGPT-25-MobileVLM_V2-1.7B")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--data-path", type=str, default="/root/autodl-tmp/Work/MME")
    parser.add_argument("--out-path", type=str, default="/root/autodl-tmp/Work/MobileVLM/mobilevlm/eval/SparseGPT-25-MobileVLM_V2-1.7B_MMEaccp.json")
    args = parser.parse_args()

    # load model
    from mobilevlm.model.mobilevlm import load_pretrained_model
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path)
    vision_tower = model.get_vision_tower()
    # if not vision_tower.is_loaded:
    vision_tower.load_model()
    vision_tower.to(device="cuda", dtype=torch.float16)

    # create GQA dataset
    MME = MMEDataset(args.data_path, image_processor, tokenizer, model.config, args.conv_mode)
    MMELoader = DataLoader(MME, batch_size=1, num_workers=4, shuffle=False)

    # load data and record
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    # cal scores
    def cal_score(pred, ans):
        return float(pred==ans)

    # open js file
    import json
    out_path = args.out_path
    f = open(out_path, "a+", encoding="utf-8")
    acc_dict = {}
    accp_dict = {}
    imacc_dict = {}
    cate_id = 0
    last_category = "code_reasoning"
    for item in tqdm(MME, total=len(MME)):
        answer = item["answer"]
        input_ids = item["input_ids"]
        image_tensor = item["image_tensor"]
        question = item["question"]
        category = item["category"]
        image_id = item["image_id"]
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
        # acc
        if acc_dict.get(category, None) is None:
            acc_dict[category] = []
        acc_dict[category].append(score)
        # acc+ 
        if imacc_dict.get(image_id, None) is None:
            imacc_dict[image_id] = int(score)
        else:
            assert(isinstance(imacc_dict[image_id], int))
            imacc_dict[image_id] = float(imacc_dict[image_id] & int(score))
            if accp_dict.get(category, None) is None:
                accp_dict[category] = []
            accp_dict[category].append(imacc_dict[image_id])

        f.write(json.dumps({
            "question_id": image_id,
            "score": score,
            "prompt": question,
            "answer": answer,
            "model_answer": outputs,
            "category": category,
            })+"\n")
    
    Cognition = set(["code_reasoning", 
                     "numerical_calculation",
                     "text_translation",
                     "commonsense_reasoning",
                     ])
    acca_list = []
    accp_list = []
    accc_list = []
    for cate in acc_dict.keys():
        f.write("Acc. on " + cate + ": " + str(sum(acc_dict[cate])/len(acc_dict[cate])) + "\t")
        f.write("Acc+. on " + cate + ":" + str(sum(accp_dict[cate])/len(accp_dict[cate])) + "\n")
        acca_list.append((sum(acc_dict[cate])/len(acc_dict[cate]) + sum(accp_dict[cate])/len(accp_dict[cate])))
        if cate in Cognition:
            accc_list.append(sum(acc_dict[cate])/len(acc_dict[cate]) + sum(accp_dict[cate])/len(accp_dict[cate]))
        else:
            accp_list.append(sum(acc_dict[cate])/len(acc_dict[cate]) + sum(accp_dict[cate])/len(accp_dict[cate]))
    f.write("Acc. Acc+. on cognition: " + str(sum(accc_list)) + "\n")
    f.write("Acc. Acc+. on perception: " + str(sum(accp_list)) + "\n")
    f.write("Acc. Acc+. on sum: " + str(sum(acca_list)) + "\n")
    f.close()