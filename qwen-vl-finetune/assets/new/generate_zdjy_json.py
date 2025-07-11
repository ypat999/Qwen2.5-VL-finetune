import os
import json

# 设置图片文件夹路径和输出json路径
image_folder = "qwen-vl-finetune/assets/new/zdjy"
output_json = "qwen-vl-finetune/assets/new/zdjy_annotations.json"

# 支持的图片格式
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 遍历图片文件夹
items = []
for fname in os.listdir(image_folder):
    if os.path.splitext(fname)[1].lower() in img_exts:
        item = {
            "image": f"{image_folder}/{fname}",
            "conversations": [
                {
                    "from": "human",
                    "value": "图片中是否存在占道经营现象？\n<image>"
                },
                {
                    "from": "gpt",
                    "value": "是"
                }
            ]
        }
        items.append(item)

# 写入json文件
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print(f"已生成标注文件：{output_json}")