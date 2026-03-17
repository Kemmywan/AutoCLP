# utils/extract_dialogue.py
# 从原始txt问诊数据中提取Dialogue段落
# python utils/extract_dialogue.py --lengthMin=500 --lengthMax=600 --total=100 --save=F

import re
import json
import os
import argparse

# ────────────────────────────────────────
# 配置项
# ────────────────────────────────────────
INPUT_FILE = 'raw/2020.txt'     # 原始txt路径

# ────────────────────────────────────────
# Step 1: 读取原始文件
# ────────────────────────────────────────
def read_raw(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# ────────────────────────────────────────
# Step 2: 按条目切分
# 每条数据以id=开头，用正则切分
# ────────────────────────────────────────
def split_records(raw_text: str) -> list:
    # 以"id="作为每条数据的起始标志切分
    records = re.split(r'(?=^id=)', raw_text, flags=re.MULTILINE)
    # 过滤空块
    records = [r.strip() for r in records if r.strip()]
    print(f'[解析] 共识别到 {len(records)} 条原始数据。')
    return records

# ────────────────────────────────────────
# Step 3: 从单条数据中提取id和Dialogue内容
# ────────────────────────────────────────
def parse_record(record: str) -> dict | None:
    # 提取id
    id_match = re.search(r'^id=(.+)$', record, re.MULTILINE)
    if not id_match:
        return None
    record_id = id_match.group(1).strip()

    # 提取Dialogue段落
    dialogue_match = re.search(
        r'^Dialogue\s*\n(.*?)(?=\n\s*\n|\Z)',
        record,
        re.MULTILINE | re.DOTALL
    )
    if not dialogue_match:
        return None

    dialogue = dialogue_match.group(1).strip()
    return {
        'id': record_id,
        'dialogue': dialogue
    }

# ────────────────────────────────────────
# Step 4: 过滤长度 > MIN_LENGTH 的条目
# ────────────────────────────────────────
def filter_by_length(parsed_list: list, min_len: int, max_len: int) -> list:
    filtered = [
        item for item in parsed_list
        if (len(item['dialogue']) > min_len) and (len(item['dialogue']) < max_len)
    ]
    print(f'[过滤] 长度在{min_len}字～{max_len}字之间的条目：{len(filtered)} 条（原始有效条目：{len(parsed_list)} 条）')
    return filtered

# ────────────────────────────────────────
# Step 5: 输出到json
# ────────────────────────────────────────
def save_json(data: list, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'[输出] 已保存 {len(data)} 条数据至 {output_path}')

# ────────────────────────────────────────
# 主流程
# ────────────────────────────────────────
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="extract_dialogue"
    )
    parser.add_argument(
        "-lmi","--lengthMin",
        type=int,
        required=True,
        help="请输对话长度最小值",  
    )
    parser.add_argument(
        "-lma", "--lengthMax",
        type=int,
        required=True,
        help="请输入对话长度最大值",
    )
    parser.add_argument(
        "-t", "--total",
        type=int,
        required=True,
        help="请输入要提取对话段数的数量"
    )
    parser.add_argument(
        "-s", "--save",
        choices=["T", "F"],
        required=True,
        help="是否保存(T/F)"
    )
    args = parser.parse_args()

    print('=' * 40)
    print('[extract_dialogues.py] 问诊数据提取与过滤')
    print('=' * 40)

    raw = read_raw(INPUT_FILE)
    records = split_records(raw)

    parsed = []
    for r in records:
        result = parse_record(r)
        if result:
            parsed.append(result)

    print(f'[解析] 成功解析 {len(parsed)} 条有效数据。')

    filtered = filter_by_length(parsed, args.lengthMin, args.lengthMax)

    if len(filtered) > args.total:
        print(f"From {len(filtered)} extract {args.total} records")
        final = filtered[:args.total]
    else:
        print(f"Only {len(filtered)} records, no more than {args.total}")
        final = filtered

    if args.save == "T":
        if len(filtered) > 0:
            save_json(final, f"test_data/2020_{args.lengthMin}_{args.lengthMax}_{args.total}.txt")
        else:
            print("No records!")

