#!/bin/bash

export WANDB_PROJECT="question-answering"

RUN_NAME="run_mqa_ssl"
OUTPUT_DIR="/home/ray_chen/CKPT/${RUN_NAME}"


TRAIN_DATASET="duorc_p duorc_s hotpotqa nq quac squad triviaqa wikihop"
EVAL_DATASET="boolq complexq comqa newsqa duorc_p duorc_s hotpotqa nq quac squad triviaqa wikihop"


python3 run_qa_no_trainer_meta.py \
    --type origin \
    --loss meta-sim \
    --multi_dataset \
    --k_support 5 \
    --k_query 5 \
    --dataset_name dummy \
    --train_dataset_names "$TRAIN_DATASET" \
    --eval_dataset_names "$EVAL_DATASET" \
    --report_to wandb \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --model_name_or_path bert-base-uncased \
    --preprocessing_num_workers 8 \
    --version_2_with_negative \
    --do_train \
    --do_eval_all \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 5 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --pad_to_max_length \
    --doc_stride 128 \
    --output_dir $OUTPUT_DIR
