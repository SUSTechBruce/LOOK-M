#!/bin/bash
EXP="llava_real_drop-0.1_0.1"
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /home/wza/Work/Bench/POPE/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${EXP}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /home/wza/Work/Bench/POPE/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/${EXP}.jsonl
