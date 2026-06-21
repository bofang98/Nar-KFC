#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc-per-node=8 --master_port=29507 run.py \
--data Video-MME \
--model llava_video_qwen2_7b \
--nframe 64 \
--verbose \
--use-subtitle


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc-per-node=8 --master_port=29507 run.py \
--data Video-MME \
--model llava_video_qwen2_7b \
--nframe 64 \
--verbose


# export AUTO_SPLIT=1

# torchrun --nproc-per-node=1 run.py \
# --data Video-MME \
# --model llava_video_qwen2_72b \
# --nframe 64 \
# --verbose \
# --use-subtitle


# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 run.py \
# --data Video-MME \
# --model llava_video_qwen2_72b \
# --nframe 64 \
# --verbose