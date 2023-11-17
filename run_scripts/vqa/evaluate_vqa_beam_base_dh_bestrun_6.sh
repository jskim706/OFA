#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8182

bpe_dir=utils/BPE
user_dir=ofa_module

# val or test
split=test
#test score : 0.2126
#val score : 0.7674

data_dir=/data/vqa/vqa_data
data=${data_dir}/vqa_${split}.tsv
ans2label_file=/data/vqa/vqa_data/trainval_ans2label.pkl
path=/home/jskim/Projects/OFA/vqa_checkpoints_lora_1116/10_0.04_5e-5_480_stage1/checkpoint_best.pt

result_path=results/vqa_${split}_beam_vqa_checkpoints_lora_111610_0.04_5e-5_480_stage1
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
    --batch-size=8 \
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