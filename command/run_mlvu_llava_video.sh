#!/usr/bin/env bash
set -euo pipefail

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# torchrun --nproc-per-node=8 --master-port=29501 run.py \
# --data MLVU_MCQ \
# --model llava_video_qwen2_7b \
# --nframe 64 \
# --verbose


export AUTO_SPLIT=1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 run.py \
--data MLVU_MCQ \
--model llava_video_qwen2_72b \
--nframe 64 \
--verbose


