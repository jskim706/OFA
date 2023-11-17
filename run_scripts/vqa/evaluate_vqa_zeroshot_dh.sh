#!/usr/bin/env bash

# This script evaluates pretrained OFA-Large checkpoint on zero-shot open-domain VQA task.

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8082

bpe_dir=utils/BPE
user_dir=ofa_module

# val or test
split=test

data_dir=/data/vqa/vqa_data
data=${data_dir}/vqa_${split}.tsv
path=checkpoints/ofa_base.pt
result_path=results/vqa_${split}_zeroshot
selected_cols=0,5,2,3,4

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=vqa_gen \
    --selected-cols=${selected_cols} \
    --bpe-dir=${bpe_dir} \
    --patch-image-size=480 \
    --prompt-type='none' \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --beam-search-vqa-eval \
    --zero-shot \
    --beam=20 \
    --unnormalized \
    --temperature=1.0 \
    --num-workers=0