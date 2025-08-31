#!/bin/bash

dataset_name="documents"
upload_user="Qwen"
model_type="Qwen3-30B-A3B-Instruct-2507"
revision="main"
left_padding=True
max_new_tokens=1024
do_sample=True
temperature=0.6
top_p=0.95
top_k=20

python main.py mode=generate \
    dataset_name=$dataset_name \
    upload_user=$upload_user \
    model_type=$model_type \
    revision=$revision \
    left_padding=$left_padding \
    max_new_tokens=$max_new_tokens \
    do_sample=$do_sample \
    generation_config.temperature=$temperature \
    generation_config.top_p=$top_p \
    generation_config.top_k=$top_k
