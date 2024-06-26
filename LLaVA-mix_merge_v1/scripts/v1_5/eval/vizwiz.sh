#!/bin/bash
# export CUDA_VISIBLE_DEVICES=4,5,6,7
MODEL_NAME="llava-kv-merge-150-0.1-0.1-0.1-0.9"
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /home/wza/Work/Bench/Vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${MODEL_NAME}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${MODEL_NAME}.json
