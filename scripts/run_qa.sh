#!/bin/bash

export WANDB_PROJECT="question-answering"

DATASET="squad"
RUN_NAME="squad-baseline"
OUTPUT_DIR="/home/ray_chen/CKPT/${RUN_NAME}"


python3 run_qa.py \
    --fp16 \
    --save_steps 5000 \
    --report_to wandb \
    --run_name ${RUN_NAME} \
    --model_name_or_path bert-base-uncased \
    --dataset_name ${DATASET} \
    --preprocessing_num_workers 8 \
    --version_2_with_negative \
    --do_train \
    --do_eval \
    --do_eval_all \
    --per_device_train_batch_size 24 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ${OUTPUT_DIR}


# python3 run_qa.py \
#     --max_train_samples 256 \
#     --max_eval_samples 24 \
#     --report_to wandb \
#     --run_name ${RUN_NAME} \
#     --model_name_or_path bert-base-uncased \
#     --dataset_name ${DATASET} \
#     --preprocessing_num_workers 8 \
#     --version_2_with_negative \
#     --do_train \
#     --do_eval \
#     --do_eval_all \
#     --per_device_train_batch_size 12 \
#     --learning_rate 3e-5 \
#     --num_train_epochs 2 \
#     --max_seq_length 384 \
#     --doc_stride 128 \
#     --output_dir ${OUTPUT_DIR}

