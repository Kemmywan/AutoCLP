"""
从 test_data/ 目录下的 2020_300_400_20.txt 到 2020_2200_2300_20.txt，
每个文件随机取出 3 个不同的 dialogue 例子，共 60 个，
写入 test_data/data_for_exp_1.txt（JSON 数组格式）。

用法: python utils/sample_dialogues.py
"""

import json
import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")

# 从 2020_300_400_20.txt 到 2020_2200_2300_20.txt（步长100，共20个文件）
FILE_RANGES = [
    (s, s + 100) for s in range(300, 2300, 100)
]

SAMPLES_PER_FILE = 1
SEED = 42


def main():
    random.seed(SEED)
    all_samples = []

    for start, end in FILE_RANGES:
        filename = f"2020_{start}_{end}_20.txt"
        filepath = os.path.join(TEST_DATA_DIR, filename)

        if not os.path.exists(filepath):
            print(f"[警告] 文件不存在，跳过: {filename}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            dialogues = json.load(f)

        if len(dialogues) < SAMPLES_PER_FILE:
            print(f"[警告] {filename} 只有 {len(dialogues)} 条，全部选取")
            selected = dialogues
        else:
            selected = random.sample(dialogues, SAMPLES_PER_FILE)

        for item in selected:
            all_samples.append({
                "source_file": filename,
                "id": item["id"],
                "dialogue": item["dialogue"],
            })

    output_path = os.path.join(TEST_DATA_DIR, "data_for_exp_1.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"已生成 {len(all_samples)} 个样本 → {output_path}")


if __name__ == "__main__":
    main()
