source /picassox/intelligent-l20x-cpfs/segmentation/tyy1/soft/env/qwen3vl/bin/activate

echo "begain run:"


CUDA_VISIBLE_DEVICES=0,3,4,5 accelerate launch \
  --num_processes 4 \
  --multi_gpu \
  train.py \
  --config ./train_configs/train_lora_v2.yaml
