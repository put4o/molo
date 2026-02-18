import json

# 读取failed_samples.json获取id列表
with open('/gz-data/analyze/failed_samples.json', 'r', encoding='utf-8') as f:
    failed_samples = json.load(f)

# 提取所有的id
failed_ids = {sample['id'] for sample in failed_samples}
print(f"从failed_samples.json中提取了 {len(failed_ids)} 个id")

# 读取beamsearch结果文件
with open('/gz-data/results/MMLong/QwenVL-7B/beamsearch_LoRA_top3.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 过滤出id在failed_ids中的条目
matched_samples = [sample for sample in results if sample['id'] in failed_ids]
print(f"在beamsearch_LoRA_top3_scored.json中找到了 {len(matched_samples)} 个匹配的样本")

# 保存结果到指定目录
output_path = '/gz-data/analyze/matched_failed_samples.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(matched_samples, f, indent=2, ensure_ascii=False)

print(f"结果已保存到: {output_path}")
