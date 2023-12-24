<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Domain-Adaptive Question Answering

This folder contains several scripts that showcase how to fine-tune a 🤗 Transformers model on a question answering dataset,
like SQuAD. 

## Prepare data
1. Download the tar data `eqa_datasets.tar.gz` from [this url](https://drive.google.com/file/d/1JmZoPckw9Nxek-TA5q4p7SrIa-KOv_as/view?usp=sharing).
2. Untar the data by the command `tar -xzvf eqa_datasets.tar.gz`.
3. Change the `ROOT_DIR` variable in `loaders/eqa_loader.py` to `path/to/your/eqa_datasets`.

## Training script for `run_qa.py`
* The sample running script can be found in [`scripts/sample.sh`](scripts/sample.sh).
* The `--model_type` can be `origin`, `cls`, or `sim`.
* For `--model_type sim`, one can further specify the SimCSE model version by setting `--simcse_name princeton-nlp/some-other-model`.

```bash
#!/bin/bash

export WANDB_PROJECT="question-answering"

RUN_NAME="sample-run"
WANDB_ENTITY="speech-lab"
OUTPUT_DIR="/tmp/debug"

TRAIN_DATASET="duorc_p duorc_s hotpotqa nq quac squad triviaqa wikihop"
EVAL_DATASET="boolq complexq comqa newsqa"


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

```

## Training scripts for `run_qa_no_trainer.py`
* The sample running script for `run_qa_no_trainer.py`, which have the same function with `run_qa.py`
* The arg `--model_type` is renamed to `--type`

```bash
#!/bin/bash

export WANDB_PROJECT="question-answering"

RUN_NAME="sample-run"
WANDB_ENTITY="speech-lab"
OUTPUT_DIR="/tmp/debug"

TRAIN_DATASET="duorc_p duorc_s hotpotqa nq quac squad triviaqa wikihop"
EVAL_DATASET="boolq complexq comqa newsqa"


python3 run_qa_no_trainer.py \
    --type origin \
    --multi_dataset \
    --max_train_samples 24 \
    --max_eval_samples 24 \
    --dataset_name dummy \
    --train_dataset_names "$TRAIN_DATASET" \
    --eval_dataset_names "$EVAL_DATASET" \
    --model_name_or_path bert-base-uncased \
    --preprocessing_num_workers 8 \
    --version_2_with_negative \
    --do_train \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTPUT_DIR
```

## Training script for `run_qa_no_trainer_meta.py`
* The support and query shot can be changed in `--k_support` and `--k_query`.
* The `--type origin` and `--loss meta-sim` is implemented (Meta-T3 with SimCSE loss).

```bash
#!/bin/bash

export WANDB_PROJECT="question-answering"

RUN_NAME="sample-run"
WANDB_ENTITY="speech-lab"
OUTPUT_DIR="/tmp/debug"

TRAIN_DATASET="duorc_p duorc_s hotpotqa nq quac squad triviaqa wikihop"
EVAL_DATASET="boolq complexq comqa newsqa"


python3 run_qa_no_trainer_meta.py \
    --type origin \
    --loss meta-sim \
    --multi_dataset \
    --max_train_samples 24 \
    --max_eval_samples 24 \
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
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 5 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTPUT_DIR
```
