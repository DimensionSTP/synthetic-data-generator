#!/bin/bash

data_type="conversational"

dataset_names=(
    "aime24"
    "arc_challenge"
    "arc_easy"
    "gpqa_diamond"
    "gpqa_extended"
    "gpqa_main"
    "gsm8k"
    "humaneval"
    "math500"
    "mbpp"
    "mmlu"
    "mmlu_pro"
    "open_rewrite_eval"
    "hrm8k_gsm8k"
    "hrm8k_ksm"
    "hrm8k_math"
    "hrm8k_mmmlu"
    "hrm8k_omni_math"
    "kmmlu"
    "haerae"
)

is_preprocessed=False
upload_user="Qwen"

model_types=(
    "Qwen3-32B"
    "Qwen3-30B-A3B"
    "Qwen3-235B-A22B"
)

revision="main"
left_padding=True
max_length=2048
max_new_tokens=256
do_sample=True
temperature=0.6
top_p=0.95
top_k=20

for model_type in "${model_types[@]}"
do
    for dataset_name in "${dataset_names[@]}"
    do
        echo "Running evaluation on dataset: $dataset_name"

        python main.py mode=test_vllm \
            data_type=$data_type \
            dataset_name=$dataset_name \
            is_preprocessed=$is_preprocessed \
            upload_user=$upload_user \
            model_type=$model_type \
            revision=$revision \
            left_padding=$left_padding \
            max_length=$max_length \
            max_new_tokens=$max_new_tokens \
            do_sample=$do_sample \
            generation_config.temperature=$temperature \
            generation_config.top_p=$top_p \
            generation_config.top_k=$top_k
    done
done
