#!/usr/bin/env python3
"""
根据 failed_samples.json 中的 id，从 beamsearch_LoRA_top3_scored.json 中提取匹配的记录
"""

import json
import os

# 文件路径
failed_samples_path = "/gz-data/analyze/failed_samples.json"
results_path = "/gz-data/results/MMLong/QwenVL-7B/beamsearch_LoRA_top3_scored.json"
output_dir = "/gz-data/analyze/"

# 读取 failed_samples.json 获取所有 id
print("正在读取 failed_samples.json...")
with open(failed_samples_path, 'r', encoding='utf-8') as f:
    failed_samples = json.load(f)

# 提取所有 id
failed_ids = set()
for sample in failed_samples:
    if 'id' in sample:
        failed_ids.add(sample['id'])

print(f"在 failed_samples.json 中找到 {len(failed_ids)} 个唯一的 id")

# 读取 beamsearch_LoRA_top3_scored.json
print("正在读取 beamsearch_LoRA_top3_scored.json...")
with open(results_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

# 筛选匹配的记录
matched_records = []
for record in results:
    if 'id' in record and record['id'] in failed_ids:
        matched_records.append(record)

print(f"在 beamsearch_LoRA_top3_scored.json 中找到 {len(matched_records)} 条匹配的记录")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 保存结果
output_path = os.path.join(output_dir, "matched_records.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(matched_records, f, indent=2, ensure_ascii=False)

print(f"结果已保存到: {output_path}")

# 显示一些统计信息
print("\n=== 统计信息 ===")
print(f"失败样本中的唯一 id 数量: {len(failed_ids)}")
print(f"匹配的记录数量: {len(matched_records)}")
print(f"匹配率: {len(matched_records)/len(failed_ids)*100:.2f}%")

# 显示一些匹配的示例
if matched_records:
    print("\n=== 前 3 条匹配记录示例 ===")
    for i, record in enumerate(matched_records[:3]):
        print(f"\n记录 {i+1}:")
        print(f"  id: {record.get('id', 'N/A')}")
        print(f"  question: {record.get('question', 'N/A')[:80]}...")
        print(f"  pred_ans: {record.get('pred_ans', 'N/A')}")
        print(f"  answer: {record.get('answer', 'N/A')}")
