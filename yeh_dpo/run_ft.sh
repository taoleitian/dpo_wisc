#!/bin/bash
# Please run this script under ${project_id} in project directory of

# Parses arguments
export HF_HOME=/nobackup2/taoleitian/HF
deepspeed_args="--master_port=11287 --include=localhost:0,1,2,3,4,5,6,7"      # Default argument
project_dir='/u/l/e/leitiantao/private/LMFlow'

cd ${project_dir}
exp_id=Meta-Llama-8b_sft
model_name_or_path=meta-llama/Meta-Llama-3-8B
#dataset_path=/u/l/e/leitiantao/private/LMFlow/data/weak2strong/sft/rm_opt_125m_w2s
#dataset_path=/u/l/e/leitiantao/private/LMFlow/data/weak2strong/sft/gt_gt_opt_125m
dataset_path=/u/l/e/leitiantao/private/LMFlow/data/hh/clean/opt_prompt/sft/all
output_dir=/nobackup2/taoleitian/neurips/hh/model/sft/${exp_id}


deepspeed ${deepspeed_args} \
  wss_hh/yeh_dpo/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-5 \
    --block_size 512 \
    --use_lora 0 \
    --deepspeed configs/ds_config_zero2.json \
    --fp16 \
    --run_name ${exp_id} \
    --logging_steps 20 \
    --do_train \
    --dataloader_num_workers 8\
    --seed 12 \
    --save_steps 999999 \
    --lr_scheduler_type constant \


