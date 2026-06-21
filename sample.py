


# Demo
from vlmeval.config import supported_VLM
from vlmeval.dataset import SUPPORTED_DATASETS

print(SUPPORTED_DATASETS)
print(supported_VLM)

import pdb; pdb.set_trace()
model = supported_VLM['Qwen2-VL-2B-Instruct']()
# Forward Single Image
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # The image features a red apple with a leaf on it.
# Forward Multiple Images
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # There are two apples in the provided images.



# DocVQA_VAL', 'DocVQA_TEST',  'ChartQA_TEST',
# python run.py --data DocVQA_TEST --model Qwen2-VL-2B-Instruct --verbose

# evaluate videos
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=1 run.py --data Video-MME --model Qwen2-VL-2B-Instruct --nframe 8 --verbose
# --nframe (int, default to 8): 从视频中采样的帧数，仅对视频多模态评测集适用
# --pack (bool, store_true): 一个视频可能关联多个问题，如 pack==True，将会在一次询问中提问所有问题
# CUDA_VISIBLE_DEVICES=7 python run.py --data MMBench-Video --model InternVL2-8B --nframe 8 --verbose
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 run.py --data Video-MME --model InternVL2-8B --nframe 8 --verbose
