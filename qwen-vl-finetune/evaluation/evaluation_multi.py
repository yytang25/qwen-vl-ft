import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from multiprocessing import Process
import math
import glob
import multiprocessing as mp
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

mp.set_start_method("spawn", force=True)
WORKERS_PER_GPU = 1


PROCESSOR_NAME = "./models/Qwen3-VL-4B-Instruct"

# MODEL_NAME = "./models/Qwen3-VL-4B-Instruct"
MODEL_NAME = "./output_singleimages/checkpoint-1"

INPUT_JSON = "./demo/single_images_test.json"

OUTPUT_JSON = "./demo/single_images_test_res_sample.json"

IMAGES_PATH = "./demo/images/"

MAX_NEW_TOKENS = 512




def resize_keep_aspect(image_path, max_size=2048):
    img = Image.open(image_path)

    w, h = img.size

    # 如果已经符合要求，直接返回
    if max(w, h) <= max_size:
        return img

    # 计算缩放比例
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img = img.resize((new_w, new_h), Image.BILINEAR)

    return img



# ================== Model ==================
class DemoServer:
    def __init__(self, gpu_id):
        print(f"🚀 Loading model on GPU {gpu_id}...")

        self.device = torch.device(f"cuda:{gpu_id}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)

        print(f"✅ Model loaded on GPU {gpu_id}")


    def infer_one(self, image_path, prompt):
        img = resize_keep_aspect(image_path, 2048)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.2,
                top_p=0.9
            )

        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        if "assistant" in output:
            output = output.split("assistant")[-1].strip()

        return output


# ================== Prompt ==================
def build_prompt(item):

    criteria = item["criteria"]
    question = item["question"]
    options = item["options"]

    criteria_text = "\n".join([
        f"{k}: level={v['level']}"
        for k, v in criteria.items()
    ])

    options_text = "\n".join([
        f"{k}. {v}"
        for k, v in options.items()
    ])

    prompt = f"""
You are an expert visual aesthetics evaluator.

You MUST analyze the image and output STRICT JSON ONLY.

---

TASKS:
1. Predict overall aesthetic score (1–100) based on visual evidence
2. Predict each criterion level: Good / Medium / Poor
3. Answer the multiple-choice question (A/B/C/D)

---

IMAGE CRITERIA:
{criteria_text}

---

QUESTION:
{question}

---

OPTIONS:
{options_text}

---

OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "total_score": "<integer 1-100 inferred from image>",
  "criteria": {{
    "Color Harmony": "<Good|Medium|Poor>",
    "Visual Style Consistency": "<Good|Medium|Poor>",
    "Sharpness": "<Good|Medium|Poor>",
    "Light and Shadow Modeling": "<Good|Medium|Poor>",
    "Creativity and Originality": "<Good|Medium|Poor>",
    "Exposure Control": "<Good|Medium|Poor>",
    "Application of Classical Composition Principles": "<Good|Medium|Poor>",
    "Depth of Field and Layering": "<Good|Medium|Poor>",
    "Visual Center Stability": "<Good|Medium|Poor>",
    "Visual Flow Guidance": "<Good|Medium|Poor>",
    "Structural Support Stability": "<Good|Medium|Poor>",
    "Appropriateness of Negative Space": "<Good|Medium|Poor>",
    "Subject Integrity": "<Good|Medium|Poor>"
  }},
  "answer": "<A|B|C|D>"
}}

---

IMPORTANT RULES:

- DO NOT copy any fixed value pattern.
- DO NOT output identical labels for all criteria.
- Each criterion MUST be judged independently from image evidence.
- total_score MUST NOT be a template value; it must reflect real visual quality.
- answer MUST be grounded in visible evidence.
- If unsure, choose Medium instead of guessing Good/Poor blindly.

---

Return ONLY valid JSON. No explanation. No markdown.
"""

    return prompt


# ================== JSON parser ==================
def extract_json(text):
    try:
        text = text.replace("```json", "").replace("```", "")
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(text[start:end + 1])
    except:
        return None


# ================== NEW：读取历史结果 ==================
def build_done_set_from_parts():

    files = glob.glob(OUTPUT_JSON + ".part*.json")

    done = set()

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                for x in data:
                    done.add(x["image_path"])
        except:
            continue

    print(f"♻️ Found done samples: {len(done)}")

    return done


# ================== Worker ==================
def worker_run(worker_id, gpu_id, data_chunk):

    server = DemoServer(gpu_id)

    out_path = OUTPUT_JSON + f".part{worker_id}.json"

    # ====== 只加载已有结果（不再做过滤）=====
    results = []

    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"♻️ worker {worker_id} load existing: {len(results)}")
        except:
            results = []

    for idx, item in enumerate(tqdm(data_chunk, desc=f"worker-{worker_id}")):

        image_path = IMAGES_PATH + item["image_path"]

        if not os.path.exists(image_path):
            continue

        prompt = build_prompt(item)

        try:
            raw = server.infer_one(image_path, prompt)
            parsed = extract_json(raw)

            if not parsed:
                continue

            record = {
                "image_path": item["image_path"],
                "total_score": parsed.get("total_score"),
                "criteria": parsed.get("criteria"),
                "question": item.get("question"),
                "options": item.get("options"),
                "answer": parsed.get("answer")
            }

            results.append(record)

            # ====== 每条都写 ======
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            os.replace(tmp_path, out_path)

        except Exception as e:
            print(f"error worker {worker_id}:", e)
            continue

    print(f"✅ worker {worker_id} done, total: {len(results)}")


# ================== Merge ==================
def merge_results():

    files = glob.glob(OUTPUT_JSON + ".part*.json")

    unique = {}

    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            for x in json.load(fp):
                unique[x["image_path"]] = x

    all_results = list(unique.values())

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"✅ merged done: {len(all_results)}")


# ================== Main ==================
def main():

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ====== 🔥 核心：读取历史结果 ======
    done_set = build_done_set_from_parts()

    # ====== 🔥 过滤未完成数据 ======
    new_data = []
    for item in data:
        image_path = item["image_path"]
        if image_path not in done_set:
            new_data.append(item)

    print(f"🚀 remaining to process: {len(new_data)}")

    data = new_data

    NUM_GPUS = torch.cuda.device_count()
    

    TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU

    print(f"🚀 GPUs: {NUM_GPUS}, total workers: {TOTAL_WORKERS}")

    if len(data) == 0:
        print("✅ nothing to process")
        merge_results()
        return

    chunk_size = math.ceil(len(data) / TOTAL_WORKERS)

    chunks = [
        data[i:i + chunk_size]
        for i in range(0, len(data), chunk_size)
    ]

    processes = []

    for i, chunk in enumerate(chunks):

        gpu_id = i // WORKERS_PER_GPU

        p = Process(
            target=worker_run,
            args=(i, gpu_id, chunk)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    merge_results()


# ================== Entry ==================
if __name__ == "__main__":
    main()


"""
CUDA_VISIBLE_DEVICES=0,3,5,7 python evaluation/evaluation_multi.py

"""