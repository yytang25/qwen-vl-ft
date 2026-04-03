import json

# =========================
# score -> level
# =========================
def score_to_level(score):
    if score < 5:
        return "Poor"
    elif score < 7:
        return "Medium"
    else:
        return "Good"


# =========================
# optional: simplify reason
# =========================
def simplify_reason(text):
    # 轻微压缩，避免过长（可删）
    return text.strip()


# =========================
# main conversion function
# =========================
def convert_item(item):
    image_path = item["image_path"]
    criteria = item["criteria"]
    total_score = item.get("total_score", None)

    # -------- build GPT output --------
    criteria_out = {}

    for k, v in criteria.items():
        score = v["score"]
        reason = simplify_reason(v["reason"])

        criteria_out[k] = {
            "level": score_to_level(score),
            "reason": reason
        }

    gpt_output = {
        "criteria": criteria_out,
        "total_score": total_score
    }

    # -------- instruction prompt --------
    human_prompt = (
        "You are an aesthetics expert. Please evaluate the image across the following 13 sub-criteria:\n"
        "Color Harmony, Visual Style Consistency, Sharpness, Light and Shadow Modeling, "
        "Creativity and Originality, Exposure Control, Application of Classical Composition Principles, "
        "Depth of Field and Layering, Visual Center Stability, Visual Flow Guidance, "
        "Structural Support Stability, Appropriateness of Negative Space, Subject Integrity.\n\n"
        "For each criterion, output a level (Poor / Medium / Good) with a brief reason.\n"
        "Finally provide a total score.\n\n"
        "<image>"
    )

    return {
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": human_prompt
            },
            {
                "from": "gpt",
                "value": json.dumps(gpt_output, ensure_ascii=False, indent=2)
            }
        ]
    }


# =========================
# batch processing
# =========================
def convert_dataset(input_json, output_json):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in data:
        results.append(convert_item(item))

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_json}")


# =========================
# example usage
# =========================
if __name__ == "__main__":
    input_path = "./demo/single_images.json"      # 你的原始数据
    output_path = "./demo/single_images_train_convert.json"    # 转换后数据

    convert_dataset(input_path, output_path)
