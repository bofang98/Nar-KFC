#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc-per-node=8 run.py \
--data LongVideoBench \
--model llava_video_qwen2_7b \
--nframe 64 \
--verbose


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# torchrun --nproc-per-node=8 run.py \
# --data LongVideoBench \
# --model llava_video_qwen2_7b \
# --nframe 16 \
# --verbose


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# torchrun --nproc-per-node=8 run.py \
# --data LongVideoBench \
# --model llava_video_qwen2_7b \
# --nframe 32 \
# --verbose


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# torchrun --nproc-per-node=8 run.py \
# --data LongVideoBench \
# --model llava_video_qwen2_7b \
# --nframe 64 \
# --verbose

# --model PLLaVA-7B \
# Qwen2-VL-7B-Instruct      video_llm
# InternVL2-8B
# llava_onevision_qwen2_7b_ov