#!/bin/bash
MODEL_NAME=llava-tok-prun
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /home/wza/Work/Bench/TextVQA/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    # --model-path /root/LLaVA-PruMerge/checkpoints/llava-v1.5-7b-lora-prunemerge \
    # --model-base lmsys/vicuna-7b-v1.5 \
    # --model-path liuhaotian/llava-v1.5-7b-lora

python -m llava.eval.eval_textvqa \
    --annotation-file /home/wza/Work/Bench/TextVQA/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${MODEL_NAME}.jsonl