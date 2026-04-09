import json

def build_prompt(item):
    return f"""AES_COMP, Generate an image based on the following description.

[Content]
{item["content_description"].strip()}

[Composition]
{item["composition_analysis"].strip()}
"""

def convert_dataset(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    new_data = []
    for item in data:
        new_item = {
            "image": item["image_path"],
            "text": build_prompt(item)
        }
        new_data.append(new_item)

    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"✅ Done! Saved to {output_path}")
    print(f"Total samples: {len(new_data)}")


if __name__ == "__main__":
    # train_sample
    input_path = "./demo/single_images.json"            # input_path
    output_path = "./demo/single_images_train.json"     # output_path

    # test_sample
    input_path = "./demo/track_2_test_sample.json"            # input_path
    output_path = "./demo/track_2_test_sample_test.json"      # output_path

    convert_dataset(input_path, output_path)
