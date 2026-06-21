#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc-per-node=1 run.py \
--data Video-MME \
--model Llama-3-VILA1.5-8b \
--nframe 8 \
--verbose \
--use-subtitle

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# torchrun --nproc-per-node=8 run.py \
# --data Video-MME \
# --model Llama-3-VILA1.5-8b \
# --nframe 8 \
# --verbose





# --model PLLaVA-7B \
# Qwen2-VL-7B-Instruct      video_llm
# InternVL2-8B
# llava_onevision_qwen2_7b_ov