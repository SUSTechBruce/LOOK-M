#!/bin/bash
SPLIT="mmbench_dev_20230712"
EXP="llava_real_drop-0.1_0.1"
python -m llava.eval.model_vqa_mmbench \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /home/wza/Work/Bench/MMB/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/${EXP}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment ${EXP}
