source ./soft/env/qwen3vl/bin/activate

echo "begain run:"


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes 4 \
  --multi_gpu \
  train.py \
  --config ./train_configs/train_lora.yaml
