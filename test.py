import json
import ast

result_file = "/gz-data/results/MMLong/QwenVL-7B/beamsearch_LoRA_top3_scored.json"
output_stats = "/gz-data/analysis_ranking_errors.json"

errors_analysis = []

with open(result_file, 'r') as f:
    data = json.load(f)

for item in data:
    if not item.get("evidence_pages"):
        continue # 跳过无 GT
        
    try:
        gt_pages = set(ast.literal_eval(item["evidence_pages"]))
        rank_pages = ast.literal_eval(item["pages_ranking"])[:3]
    except:
        continue
        
    # 检查 Rank 1 是否命中
    if rank_pages[0] not in gt_pages:
        errors_analysis.append({
            "id": item["id"],
            "question": item["question"],
            "rank_1": rank_pages[0],
            "gt_pages": list(gt_pages),
            "pred_ans": item.get("pred_ans", "N/A"),
            "ground_truth": item.get("answer", "N/A"),
            "doc_id": item["doc_id"]
        })

print(f"Found {len(errors_analysis)} samples where Rank-1 is WRONG.")
# 保存前 20 个错误案例用于分析
with open(output_stats, 'w') as f:
    json.dump(errors_analysis[:20], f, indent=2, ensure_ascii=False)