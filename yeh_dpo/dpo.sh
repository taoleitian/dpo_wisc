#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4
#export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/nobackup2/taoleitian/HF
deepspeed_args="--master_port=17903 --include=localhost:0,1,2,3,7,4,5,6"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi
project_dir=$(cd "$(dirname $0)"/..; pwd)
cd /u/l/e/leitiantao/private/LMFlow/

exp_id=dpo_sft_llama3
model_name_or_path=/nobackup2/samuelyeh/reliable_rf/outputs/sft_llama3
dataset_path=/u/l/e/leitiantao/private/LMFlow/data/hh/clean/opt_prompt/train.jsonl
training_sample_number=$(wc -l < "$dataset_path")
bsz=2
num_train_epochs=1.0
temp=0.1
eval_steps=$(echo "scale=0; ($training_sample_number -100) * $num_train_epochs / $bsz / 32" | bc)
exp_name=${exp_id}
output_dir=/nobackup2/taoleitian/neurips/hh/model/ds_num/dpo/${exp_id}
log_dir=${project_dir}/log/${exp_name}


deepspeed ${deepspeed_args} \
  wss_hh/yeh_dpo/w2s_dpo.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate 5e-5 \
    --block_size 512 \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --deepspeed configs/ds_config_zero2.json \
    --bf16 \
    --run_name ${exp_name} \
    --validation_split_percentage 0 \
    --logging_steps 10 \
    --do_train \
    --use_lora 1 \
    --lora_r 16 \
    --lr_scheduler_type constant \
    --save_steps 999999 \
    --evaluation_strategy steps\
    --eval_steps ${eval_steps} \
    --weight_decay 0.0001 \
    --dataloader_num_workers 8 \
    --seed 14 \
    --temp ${temp} \

