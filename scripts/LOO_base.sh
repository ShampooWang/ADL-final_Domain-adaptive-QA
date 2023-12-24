#!/bin/bash
domain_type="in out"
corpora="comqa duorc_p duorc_s hotpotqa newsqa nq quac squad triviaqa wikihop"
#seeds=(0 1 2 3 42)

export WANDB_PROJECT="question-answering"

WANDB_ENTITY="speech-lab"

while [ 1 ]
do
    echo "Please enter your corpus."
    echo "Default corpora: ${corpora}"
    read corpus
    if [[ ${corpora} == *${corpus}* ]]; then
        break
    else
        echo "Input corpus is not in default corpora, please input again."
        continue
    fi
    echo ${corpus}
done

EVAL_DATASET=${corpus}
TRAIN_DATASET=${corpora//${corpus}/}
echo "train_corpora: ${TRAIN_DATASET}"
echo "eval_corpus: ${EVAL_DATASET}"


RUN_NAME="LOO-${corpus}"
OUTPUT_DIR="/work/jgtf0322/Domain-Adaptive-QA/checkpoints/${RUN_NAME}/"
python3 run_qa.py \
    --model_type cls \
    --multi_dataset \
    --dataset_name dummy \
    --train_dataset_names "${TRAIN_DATASET}" \
    --eval_dataset_names "${EVAL_DATASET}" \
    --report_to wandb \
    --run_name ${RUN_NAME} \
    --wandb_entity ${WANDB_ENTITY} \
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
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir false