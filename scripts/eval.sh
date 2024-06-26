#!/bin/bash

#SBATCH --job-name=eval_origin_speed     # 作业名称
#SBATCH --account=PAS2473		    # Project ID
#SBATCH --output=/users/PAS2473/brucewan666/Faster-LLaVA/MileBench/output_logs/eval_origin_speed_t-0.1-0.1.log         # 输出日志文件
#SBATCH --error=/users/PAS2473/brucewan666/Faster-LLaVA/MileBench/output_logs/eval_origin_speed_t_error-0.1-0.1.log          # 错误日志文件
#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks-per-node=1         # 每个节点的任务数
#SBATCH --cpus-per-task=4           # 每个任务使用的 CPU 核心数
#SBATCH --gpus-per-node=1	        # GPU per node
#SBATCH --mem=50G                  # 内存限制
#SBATCH --time=12:00:00             # 作业运行时间限制

# 运行命令或脚本 wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh

source $HOME/anaconda3/bin/activate /users/PAS2473/brucewan666/anaconda3/envs/milebench

export TOKENIZERS_PARALLELISM=false

GEN_SCRIPT_PATH=generate_new.py
EVAL_SCRIPT_PATH=evaluate.py
DATA_DIR=/fs/scratch/PAS2473/zhongwei_models/cache/datasets--FreedomIntelligence--MileBench/snapshots/736527eedd9ecfd94609f4936a989e4fb8e0527d/MLBench
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