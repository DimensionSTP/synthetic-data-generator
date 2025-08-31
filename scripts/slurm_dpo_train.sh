#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --partition=8gpu
#SBATCH --gres=gpu:8
#SBATCH --nodelist=gpu-8-003
#SBATCH --output=logs/train_output.log
#SBATCH --error=logs/train_error.log

cd ~/llm-fine-tune

module add compilers/cuda/12.4 compilers/gcc/10.2.0 libraries/nccl/2.21.5
source activate myenv

data_type="conversational"
split_ratio=1e-2
is_strict_split=False
dataset_name="tulu"
is_preprocessed=False
strategy="deepspeed"
upload_user="Qwen"
model_type="Qwen3-8B"
revision="main"
left_padding=True
is_enable_thinking=False
is_quantized=False
is_peft=False
max_length=2048
is_bf16=True
batch_size=8
eval_batch_size=8
gradient_accumulation_steps=2
dpo_beta=0.1
lr=5e-7
weight_decay=1e-1
warmup_ratio=5e-2
epoch=2
step=250
workers_ratio=8
use_all_workers=False

if [ "$strategy" = "deepspeed" ]; then
    deepspeed main.py --config-name=dpo.yaml mode=train \
        data_type=$data_type \
        split_ratio=$split_ratio \
        is_strict_split=$is_strict_split \
        dataset_name=$dataset_name \
        is_preprocessed=$is_preprocessed \
        strategy=$strategy \
        upload_user=$upload_user \
        model_type=$model_type \
        revision=$revision \
        left_padding=$left_padding \
        is_enable_thinking=$is_enable_thinking \
        is_quantized=$is_quantized \
        is_peft=$is_peft \
        max_length=$max_length \
        is_bf16=$is_bf16 \
        batch_size=$batch_size \
        eval_batch_size=$eval_batch_size \
        gradient_accumulation_steps=$gradient_accumulation_steps \
        dpo_beta=$dpo_beta \
        lr=$lr \
        weight_decay=$weight_decay \
        warmup_ratio=$warmup_ratio \
        epoch=$epoch \
        step=$step \
        workers_ratio=$workers_ratio \
        use_all_workers=$use_all_workers
else
    python main.py --config-name=dpo.yaml mode=train \
        data_type=$data_type \
        split_ratio=$split_ratio \
        is_strict_split=$is_strict_split \
        dataset_name=$dataset_name \
        is_preprocessed=$is_preprocessed \
        strategy=$strategy \
        upload_user=$upload_user \
        model_type=$model_type \
        revision=$revision \
        left_padding=$left_padding \
        is_enable_thinking=$is_enable_thinking \
        is_quantized=$is_quantized \
        is_peft=$is_peft \
        max_length=$max_length \
        is_bf16=$is_bf16 \
        batch_size=$batch_size \
        eval_batch_size=$eval_batch_size \
        gradient_accumulation_steps=$gradient_accumulation_steps \
        dpo_beta=$dpo_beta \
        lr=$lr \
        weight_decay=$weight_decay \
        warmup_ratio=$warmup_ratio \
        epoch=$epoch \
        step=$step \
        workers_ratio=$workers_ratio \
        use_all_workers=$use_all_workers
fi
