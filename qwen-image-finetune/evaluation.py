import os
from diffusers import DiffusionPipeline
import torch
import json
import shutil

use_lora = True

json_path = "./demo/track_2_test_sample_test.json"
ori_img_path = 'path/to/images/'

with open(json_path, "r") as f:
    test_images = json.load(f)

lora_step = 12000
model_name = 'path/to/Qwen/Qwen-Image'
lora_path = f'./output_lora_qwen_image/checkpoint-{lora_step}'

ori_save_path = f'./demo/ori_images'
save_path = f'./demo/results/qwen-image-lora-{lora_step}' if use_lora else f'./demo/results/qwen-image'

os.makedirs(ori_save_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

# Load LoRA weights
if use_lora:
    pipe.load_lora_weights(lora_path, adapter_name="lora")
    pipe.enable_lora()
    # pipe.set_adapters(["lora"], adapter_weights=[1.0])

for idx, image in enumerate(test_images):
    image_name = image['image']
    ori_image_path = os.path.join(ori_img_path, image_name)
    shutil.copy(ori_image_path, os.path.join(ori_save_path, image_name))
    prompt = image['text']
    if not use_lora:
        prompt = prompt.replace("AES_COMP, ", "")
    print(idx, prompt)
    negative_prompt =  " "
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=20,
        true_cfg_scale=5,
        max_sequence_length=1024,
        generator=torch.Generator(device="cuda").manual_seed(346346)
    )

    image.images[0].save(f"{save_path}/{image_name}")


