#!/bin/bash
EXP="llava_real_drop-0.1_0.1"
python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /home/wza/Work/Bench/ScienceQA/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${EXP}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${EXP}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${EXP}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${EXP}_result.json
