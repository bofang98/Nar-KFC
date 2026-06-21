#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --master_port=29501 --nproc-per-node=8 run.py \
--data LongVideoBench \
--model InternVL2-8B \
--nframe 8 \
--verbose


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# torchrun --nproc-per-node=8 run.py \
# --data LongVideoBench \
# --model InternVL2-8B \
# --nframe 16 \
# --verbose


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# torchrun --nproc-per-node=8 run.py \
# --data LongVideoBench \
# --model InternVL2-8B \
# --nframe 32 \
# --verbose


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# torchrun --nproc-per-node=8 run.py \
# --data LongVideoBench \
# --model InternVL2-8B \
# --nframe 64 \
# --verbose
