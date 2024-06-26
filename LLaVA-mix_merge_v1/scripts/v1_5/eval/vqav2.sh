#!/bin/bash
CKPT="llava-real-drop-0.1-0.1"    
SPLIT="llava_vqav2_mscoco_test-dev2015"
CHUNKS=1
IDX=0
mkdir -p ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT
python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/llava-v1.5-7b \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder /home/wza/Work/Bench/VQAv2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1

# wait

# output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

