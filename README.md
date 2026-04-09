# Qwen VL / Qwen-Image Fine-tuning

**Paper:** This repository is the **official implementation** of [PortraitCraft: A Benchmark for Portrait Composition Understanding and Generation](https://arxiv.org/abs/2604.03611) (arXiv:2604.03611).

---

This repository supports **Portrait Composition and Generation** competition-style training. It covers two tracks: training data layouts and scripts for each are in the matching subdirectory.

| Track | Task | Directory |
|-------|------|-----------|
| **Track 1: Portrait Composition Understanding** | Composition understanding and judgment (multimodal VL) | [`qwen-vl-finetune/`](qwen-vl-finetune/) |
| **Track 2: Portrait Composition Generation** | Composition-oriented image generation | [`qwen-image-finetune/`](qwen-image-finetune/) |

Use the subdirectory that matches your track for data prep, training, and evaluation.

---

## Data download

The **PortraitCraft** dataset is published on Hugging Face. Download or `datasets.load_dataset` locally, then set paths in each track’s configs and conversion scripts.

| Item | URL |
|------|-----|
| PortraitCraft dataset (train / eval for both tracks) | [https://huggingface.co/datasets/zijielou/PortraitCraft](https://huggingface.co/datasets/zijielou/PortraitCraft) |

---

## Pretrained models

Released **PortraitCraft** checkpoints live on Hugging Face. After download, point `MODEL_PATH`, `model_name_or_path`, or YAML config fields to the local directory (pick the checkpoint variant that matches your track).

| Use | URL |
|-----|-----|
| PortraitCraft pretrained weights (Track 1 & Track 2 as provided on the model card) | [https://huggingface.co/yytang225/PortraitCraft](https://huggingface.co/yytang225/PortraitCraft) |


### Requirements

You could use follow version of packages:
```bash
cd qwen-vl-ft/
pip install -r requirements.txt
```


## Track 1: `qwen-vl-finetune` (Portrait Composition Understanding)

Track 1 fine-tunes Qwen VL for composition-related understanding. The following documents the `qwenvl` layout and `qwen-vl-finetune` launch scripts.

### Workflow

1. Prepare and convert the dataset (see **Custom Dataset Configuration** below).
2. Edit model path, data, and hyperparameters in the training scripts.
3. Run inference after training and convert outputs to the required standard JSON submission format.

### Repository structure (`qwenvl`)

The `qwenvl` directory contains the following components:

#### `train/`
- `trainer.py`: Main trainer updated from Huggingface Trainer
- `train_qwen.py`: Main file for training
- `argument.py`: Dataclasses for model, data and training arguments

#### `data/`
- `__init__.py`: Contains datasets configs
- `data_processor.py`: Data processing module for QwenVL models
- `rope2d.py`: Provide RoPE implementation

#### `tools`
- `process_bbox.ipynb`: Convert bbox into QwenVL format. If you have grounding data, please refer this file to tranform your data.
- `pack_data.py`: Pack data into even length buckets.


### Custom Dataset Configuration

The customized data should have the format like:

#### JSON Data Structure

**Media Specification**:
- `image`: Contains path to the media file (required)
- Media tags in prompts:
    - `<image>` for image understanding tasks

#### Example Instances:

1. **Single Image Example**:
```json
[
    {
      "image_path": "unsplash_people_00010_67dcd75a09e4.jpg",
      "criteria": {
        "Color Harmony": {
          "score": 7.0,
          "reason": "The vibrant colors of the mural and the subjects' clothing create a lively and energetic mood, though the extreme colorfulness makes the image slightly busy."
        },
        "Visual Style Consistency": {
          "score": 7.5,
          "reason": "The overall aesthetic—bright, casual, vibrant, and sunny—is consistently maintained throughout the image, fitting the intended lifestyle mood."
        },
        "Sharpness": {
          "score": 7.6,
          "reason": "The overall image is clear, with key details like facial features, hair, and clothing of the main subjects well-defined and sharp."
        },
        "Light and Shadow Modeling": {
          "score": 5.4,
          "reason": "The hard directional sunlight creates harsh shadows under chins and noses, which is not particularly flattering for portraiture and lacks refined tonal transitions."
        },
        "Creativity and Originality": {
          "score": 6.2,
          "reason": "The image feels like a very conventional lifestyle stock photograph, lacking a unique perspective or original visual language."
        },
        "Exposure Control": {
          "score": 6.3,
          "reason": "Exposure is generally reasonable, but the bright, direct sunlight causes some harsh highlights on the skin and clothing, with deep shadows that slightly reduce tonal layering."
        },
        "Application of Classical Composition Principles": {
          "score": 5.0,
          "reason": "The composition is a standard group lineup, but the framing feels somewhat arbitrary, particularly on the right side where elements are poorly cropped."
        },
        "Depth of Field and Layering": {
          "score": 5.2,
          "reason": "There is minimal depth of field; the brightly colored, busy background is almost entirely in focus, which fails to provide good separation for the subjects."
        },
        "Visual Center Stability": {
          "score": 5.1,
          "reason": "While the interacting women in the center draw the eye, the highly distracting background and the confusing elements on the right edge weaken the stability of the visual center."
        },
        "Visual Flow Guidance": {
          "score": 5.2,
          "reason": "The viewer's eye naturally follows the smiles and the glasses, but the flow is interrupted by the cluttered background and the awkward framing on the right."
        },
        "Structural Support Stability": {
          "score": 5.1,
          "reason": "The group forms a loose horizontal arrangement that grounds the image moderately well, but the unresolved right edge makes the structure feel unbalanced."
        },
        "Appropriateness of Negative Space": {
          "score": 4.2,
          "reason": "The frame feels very crowded with multiple subjects and a highly complex background, lacking sufficient negative space to let the image breathe."
        },
        "Subject Integrity": {
          "score": 4.0,
          "reason": "The woman on the far right is heavily cropped and turned away, and there is a disembodied arm holding a glass entering the frame awkwardly, harming the integrity of the subjects."
        }
      },
      "total_score": 53
    }
]
```


Some examples are shown in `demo/single_images.json` and these json files could be used for training.

#### Dataset config for training

To add or modify datasets for training, follow these steps:

##### Dataset Definition Structure

1. **convert_ori_json_for_qwen3vl_train（This is just an example; you can customize the training content.）** in the format in the file `qwen-vl-finetune/convert_json_train.py`:
```python
cd qwen-vl-finetune/
python convert_json_train.py
```



2. **Create a dataset dictionary** in the format in the file `data/__init__.py`:
```python
DATASET_NAME = {
    "annotation_path": "/path/to/annotations.json",
    "data_path": "/path/to/image/data",  # Can be empty if paths are in annotations
}
```

3. **Register your dataset** by adding it to the `data_dict`:
```python
data_dict = {
    "your_dataset_name": DATASET_NAME,
    # ... other datasets
}
```

##### Sampling Rate Control

You can optionally specify sampling rates by appending `%X` to the dataset name:
- `"dataset_name%50"` will sample 50% of the data
- `"dataset_name%20"` will sample 20% of the data

##### Usage Example

1. Define your dataset:
```python
SINGLEIMAGES = {
    "annotation_path": "./demo/single_images_train_convert.json",
    "data_path": "./demo/images",
}

data_dict = {
    "my_dataset": MY_DATASET,
    "singleimages": SINGLEIMAGES,  # existing dataset
}
```

2. Use it in training:
```python
dataset_names = ["my_dataset%50"]  # Will use 50% of your dataset
configs = data_list(dataset_names)
```

##### Notes  
- The `annotation_path` should point to a JSON or JSONL file containing your dataset annotations.  
- The `data_path` can be left empty if the image paths in the annotations are absolute.  
- Sampling rates are applied per-dataset when multiple datasets are specified.  
- Some datasets you can use directly: `nyu-visionx/Cambrian-10M`, `lmms-lab/LLaVA-NeXT-Data`, `FreedomIntelligence/ALLaVA-4V`, `TIGER-Lab/VisualWebInstruct`.  
- The training data should strictly follow this format:  
  - One `<image>` tag in the question must correspond to exactly one image file  
  - Similarly, `<video>` tags must correspond to video files  
  - These special tokens should not appear in the answer text  
- For open source data that might have missing images or other issues, you can verify data completeness using `tools/check_image.py`.  


### Usage

To train a model:

```bash
#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

# ======================
# Path Configuration
# ======================
MODEL_PATH="/path/to/Qwen3-VL-4B-Instruct"  # [ModelArguments] Pretrained model path
OUTPUT_DIR="./checkpoints"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                          # [TrainingArguments] Cache directory for models

# ======================
# Model Configuration
# ======================
DATASETS="your_dataset%100"                  # [DataArguments] Dataset with sampling rate

# ======================
# Training Hyperparameters
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py \
         # Core Arguments
         --model_name_or_path $MODEL_PATH \  # [ModelArguments] Model identifier
         --tune_mm_llm True \                # [TrainingArguments] Train LLM or not
         --tune_mm_vision False \            # [TrainingArguments] Train VIT or not
         --tune_mm_mlp False \               # [TrainingArguments] Train MLP or not
         --dataset_use $DATASETS \           # [DataArguments] Dataset specification
         --output_dir $OUTPUT_DIR \          # Output directory for checkpoints
         --cache_dir $CACHE_DIR \            # [TrainingArguments] Model cache location
         
         # Precision & Memory
         --bf16 \                            # Use bfloat16 precision (Ampere+ GPUs)
         --per_device_train_batch_size 4 \   # Batch size per GPU
         --gradient_accumulation_steps 4 \   # Effective batch size multiplier
         
         # Learning Rate Configuration
         --learning_rate 2e-7 \              # Base learning rate
         --mm_projector_lr 1e-5 \            # [TrainingArguments] Projector-specific LR
         --vision_tower_lr 1e-6 \            # [TrainingArguments] Vision encoder LR
         --optim adamw_torch \               # [TrainingArguments] Optimizer selection
         
         # Sequence Configuration
         --model_max_length 4096 \           # [TrainingArguments] Max sequence length
         --data_flatten True \               # [DataArguments] Concatenate batch sequences
         --data_packing True \               # [DataArguments] Using packing data
         
         # Image Processing
         --max_pixels 576\*28\*28 \               # [DataArguments] Max image pixels (H*W) for image
         --min_pixels 16\*28\*28 \                # [DataArguments] Min image pixels for image
         # Video Processing
         --video_fps 2 \                          # [DataArguments] video fps
         --video_max_frames 8 \                   # [DataArguments] Max frames per video
         --video_min_frames 4 \                   # [DataArguments] Min frames per video
         --video_max_pixels 1664\*28\*28 \        # [DataArguments] Max pixels per video
         --video_min_pixels 256\*28\*28 \         # [DataArguments] Min pixels per video
         
         # Training Schedule
         --num_train_epochs 3 \              # Total training epochs
         --warmup_ratio 0.03 \               # LR warmup proportion
         --lr_scheduler_type "cosine" \      # Learning rate schedule
         --weight_decay 0.01 \               # L2 regularization strength
         
         # Logging & Checkpoints
         --logging_steps 10 \               # Log metrics interval
         --save_steps 500 \                 # Checkpoint save interval
         --save_total_limit 3 \             # Max checkpoints to keep

         # Lora Config
         --lora_enable True \                 # [TrainingArguments] Enable LoRA
         --lora_r 8 \                         # [TrainingArguments] LoRA r
         --lora_alpha 16 \                    # [TrainingArguments] LoRA alpha 
         --lora_dropout 0.0 \                # [TrainingArguments] LoRA dropout

         # Advanced Options
         --deepspeed zero3.json \           # DeepSpeed configuration
```

The script accepts arguments in three categories:

   - Flags to control which components to tune (`tune_mm_vision`, `tune_mm_mlp`, `tune_mm_llm`). If trained with both image and video data, tune_mm_vision should be False: `tune_mm_vision=False`
   - `data_flatten` flag means data in a batch are concat into one sequence
   - `data_packing` requires preprocess with `tools/pack_data.py`
   - Training hyperparameters, the suggested learning rate is from 1e-6 to 2e-7
   - Training resolution is critical for the model performances, hence `--max_pixels` and `--min_pixels` should be properly set
   - Training with Qwen2.5-VL-32B model, you should have 8 80G GPU refering to `scripts/sft_32b.sh`
   - `"_attn_implementation": "flash_attention_2",` could be add in the config.json of the model to use flash attention.
   - The Qwen3VL MoE model does not support DeepSpeed with ZeRO-3. Additionally, Hugging Face’s official implementation does not include support for load balancing loss currently.



**Training example:**

```bash
cd qwen-vl-finetune
bash scripts/sft_qwen3_4b.sh
```

**Evaluation example:**

```bash
cd qwen-vl-finetune
python evaluation/evaluation_multi.py
```

**Convert to the final required submission format:**

```bash
cd qwen-vl-finetune
python convert_json_test.py
```

---

## Track 2: `qwen-image-finetune` (Portrait Composition Generation)

Track 2 fine-tunes Qwen-Image for generation (e.g. LoRA or full fine-tuning). It is independent from Track 1; work under [`qwen-image-finetune/`](qwen-image-finetune/).

### Workflow

1. Prepare data and convert to the competition format (example below).
2. Update paths and hyperparameters in `train_configs` and launch commands.
3. Run evaluation after training and export results as required.

### Examples

1. **`convert_ori_json_for_qwen-image_train`** (example only; customize as needed):

```bash
cd qwen-image-finetune/
python convert_json_train.py
```

2. **Train**:

```bash
cd qwen-image-finetune/
CUDA_VISIBLE_DEVICES=7 accelerate launch train.py --config ./train_configs/train_lora.yaml
```

3. **Evaluation**:

```bash
cd qwen-image-finetune/
python evaluation.py
```

---

## Reference code

- **Track 1:** [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- **Track 2:** [FlyMyAI/flymyai-lora-trainer](https://github.com/FlyMyAI/flymyai-lora-trainer)
