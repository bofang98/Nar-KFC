#!/usr/bin/env bash
set -euo pipefail

export AUTO_SPLIT=1 

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python run.py \
--data Video-MME \
--model InternVL2-40B \
--nframe 16 \
--verbose