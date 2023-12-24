#!/bin/bash
domain_type="in out"
corpora="comqa duorc_p duorc_s hotpotqa newsqa nq quac squad triviaqa wikihop"
seeds=(0 1 2)

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

while [ 1 ]
do
    echo "Please enter your domain type (in/out)."
    read domain
    if [[ ${domain_type} == *${domain}* ]]; then
        break
    else
        echo "Wrong domain type, please input again."
        continue
    fi
done

EVAL_DATASET=${corpus}
if [ ${domain} == "in" ]; then
    TRAIN_DATASET=${corpus}
else
    TRAIN_DATASET=${corpora//${corpus}/}
fi
train_num=(5 10 100 1000 2000)


echo "exp_domain: ${domain}"
echo "train_corpora: ${TRAIN_DATASET}"
echo "eval_corpus: ${EVAL_DATASET}"

for num in ${train_num[*]}
do
    for seed in ${seeds[*]}  
    do

        RUN_NAME="${corpus}-${domain}-${num}-seed${seed}"
        OUTPUT_DIR="/work/jgtf0322/Domain-Adaptive-QA/checkpoints/${RUN_NAME}"
        echo "seed : ${seed}, train_num: ${num}"
        python3 run_qa.py \
            --seed ${seed} \
            --model_type origin \
            --multi_dataset \
            --max_train_samples ${num} \
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
            #--overwrite_output_dir false
    done
done