#!/bin/bash

# source $HOME/anaconda3/bin/activate /users/anaconda3/envs/milebench

export TOKENIZERS_PARALLELISM=false

GEN_SCRIPT_PATH=generate.py
EVAL_SCRIPT_PATH=evaluate.py
DATA_DIR=/path/to/MLBench
MODEL_CONFIG_PATH=configs/model_configs.yaml
gpu_num=1

KV_MODE=mean_h2o
HH_R=0.1
RECENT_R=0.1
MODEL=mean_h2o_${HH_R}_${RECENT_R}_speed
for dataset_name in ALFRED ActionLocalization ActionPrediction ActionSequence CLEVR-Change CharacterOrder CounterfactualInference DocVQA EgocentricNavigation GPR1200 IEdit ImageNeedleInAHaystack MMCoQA MovingAttribute MovingDirection MultiModalQA OCR-VQA ObjectExistence ObjectInteraction ObjectShuffle SceneTransition SlideVQA Spot-the-Diff StateChange TQA TextNeedleInAHaystack WebQA WikiVQA nuscenes; do
    # Set batch size: max(int(batch_image/n_img),1)
    if [ ${dataset_name} = "MMCoQA" ] || [ ${dataset_name} = "NeedleInAHaystack" ] || [ ${dataset_name} = "GPR1200" ]
    then
        BATCH_SIZE=1
    else
        BATCH_SIZE=24 # to be 24
    fi

    mkdir -p logs/${model}

    # Start generating
    accelerate launch --config_file ./configs/accelerate_configs.yaml \
        --main_process_port 29521  \
        --num_machines 1 \
        --machine_rank 0 \
        --num_processes ${gpu_num} \
        --deepspeed_multinode_launcher standard \
        \
        ${GEN_SCRIPT_PATH} \
        --data_dir ${DATA_DIR} \
        --dataset_name ${dataset_name}  \
        --model_name ${MODEL} \
        --output_dir outputs \
        --batch-image ${BATCH_SIZE} \
        --model_configs ${MODEL_CONFIG_PATH} \
        --overwrite \
        --kv_mode ${KV_MODE}
        # >> logs/${model}/${dataset_name}.log

    # Start evaluating
    python ${EVAL_SCRIPT_PATH} \
        --data-dir ${DATA_DIR} \
        --dataset ${dataset_name} \
        --result-dir outputs/${MODEL} \
        # >> logs/${model}/${dataset_name}.log

    # ############################## Combined to 1 image ###########################
    # # Start generating
    # accelerate launch --config_file ./configs/accelerate_configs.yaml \
    #     --main_process_port 29500  \
    #     --num_machines 1 \
    #     --machine_rank 0 \
    #     --num_processes ${gpu_num}  \
    #     --deepspeed_multinode_launcher standard \
    #     \
    #     ${GEN_SCRIPT_PATH} \
    #     --data_dir ${DATA_DIR} \
    #     --dataset_name ${dataset_name}  \
    #     --model_name ${model} \
    #     --output_dir outputs_combine_1 \
    #     --batch-image ${BATCH_SIZE} \
    #     --model_configs ${MODEL_CONFIG_PATH} \
    #     --overwrite \
    #     --combine_image 1 \
    #     > logs/${model}/${dataset_name}_combine_1.log

    # # Start evaluating
    # python ${EVAL_SCRIPT_PATH} \
    #     --data-dir ${DATA_DIR} \
    #     --dataset ${dataset_name} \
    #     --result-dir outputs_combine_1/${model} \
    #     >> logs/${model}/${dataset_name}_combine_1.log
done
# dump score_all
python score.py \
    --result-dir outputs \
    --models ${MODEL}  # models to eval