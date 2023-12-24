#!/bin/bash

export WANDB_PROJECT="question-answering"

RUN_NAME="sample-run"
WANDB_ENTITY="speech-lab"
OUTPUT_DIR="/tmp/debug"

TRAIN_DATASET="boolq complexq comqa duorc_p duorc_s hotpotqa newsqa nq quac squad triviaqa wikihop"
EVAL_DATASET="newsqa nq quac squad triviaqa wikihop"


python3 run_qa.py \
    --model_type origin \
    --multi_dataset \
    --max_train_samples 24 \
    --max_eval_samples 24 \
    --dataset_name dummy \
    --train_dataset_names "$TRAIN_DATASET" \
    --eval_dataset_names "$EVAL_DATASET" \
    --report_to wandb \
    --run_name $RUN_NAME \
    --wandb_entity $WANDB_ENTITY \
    --save_steps 5000 \
    --model_name_or_path bert-base-uncased \
    --preprocessing_num_workers 8 \
    --version_2_with_negative \
    --do_train \
    --do_eval_all \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --overwrite_output_dir \
    --output_dir $OUTPUT_DIR

