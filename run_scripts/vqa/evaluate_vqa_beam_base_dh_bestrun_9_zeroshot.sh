#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8183

bpe_dir=utils/BPE
user_dir=ofa_module

# val or test
split=test
#test score : 0.2126
#val score : 0.7674


data=/data/vqa/vqa_data/vqa_test.tsv
ans2label_file=/data/vqa/vqa_data/trainval_ans2label.pkl
path=checkpoints/ofa_base.pt

result_path=results/vqa_test_beam_ofa_base_zero
selected_cols=0,5,2,3,4
valid_batch_size=20

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=8183 evaluate.py \
    /data/vqa/vqa_data/vqa_test.tsv \
    --path=checkpoints/ofa_base.pt \
    --user-dir=ofa_module \
    --task=vqa_gen \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=test \
    --results-path=results/vqa_test_beam_ofa_base_zero \
    --fp16 \
    --ema-eval \
    --beam-search-vqa-eval \
    --beam=5 \
    --zero-shot \
    --unnormalized \
    --temperature=1.0 \
    --num-workers=0 \
    --model-overrides="{\"task\":\"vqa_gen\",\"data\":\"/data/vqa/vqa_data/vqa_test.tsv\",\"bpe_dir\":\"utils/BPE\",\"selected_cols\":\"0,5,2,3,4\",\"ans2label_file\":\"/data/vqa/vqa_data/trainval_ans2label.pkl\",\"valid_batch_size\":\"20\"}"