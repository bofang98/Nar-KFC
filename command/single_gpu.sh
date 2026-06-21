#!/usr/bin/env bash
set -euo pipefail

export AUTO_SPLIT=1 

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python run.py \
--data Video-MME \
--model InternVL2-40B \
--use-subtitle \
--nframe 16 \
--verbose