import json

input_file = "./demo/single_images_test_res_sample.json"
output_file = "./demo/single_images_test_res_sample_finel.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []

level_to_abc = {"Poor": "A", 
                "Medium": "B",
                "Good": "C"}


for item in data:
    if "criteria" not in item:
        continue

    new_criteria = {}

    for k, v in item["criteria"].items():
        new_criteria[k] = {
            "level": level_to_abc.get(v, "NOT_RES")
        }

    new_item = {
        "image_path": item.get("image_path"),
        "criteria": new_criteria,
        "total_score": item.get("total_score"),
        "question": item.get("question"),
        "options": item.get("options"),
        "answer": item.get("answer")
    }

    new_data.append(new_item)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"✅ converted: {len(new_data)} items")
