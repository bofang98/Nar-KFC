#!/usr/bin/env bash
set -euo pipefail

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# torchrun --nproc-per-node=8 run.py \
# --data Video-MME \
# --model llava_onevision_qwen2_7b_ov \
# --nframe 32 \
# --verbose \
# --use-subtitle


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# torchrun --nproc-per-node=8 run.py \
# --data Video-MME \
# --model llava_onevision_qwen2_7b_ov \
# --nframe 32 \
# --verbose



export AUTO_SPLIT=1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 run.py \
--data Video-MME \
--model llava_onevision_qwen2_72b_ov \
--nframe 32 \
--verbose