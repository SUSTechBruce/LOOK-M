#!/bin/bash
EXP="llava_kv_merge-0.1_0.1-0.1_0.9"
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /home/wza/Work/Bench/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${EXP}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment ${EXP} --data-dir /home/wza/Work/Bench/MME/MME_Benchmark_release_version

cd eval_tool

python calculation.py --results_dir answers/${EXP}
