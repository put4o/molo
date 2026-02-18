"""
错误样本RAG检索阶段表现分析
分析Base方法和Beamsearch方法在失败样本上的检索性能
"""

import json
import ast
import numpy as np
from collections import defaultdict

def parse_list_string(s):
    """解析字符串格式的列表，如 '[1, 2, 3]' """
    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except:
            return []
    return s if isinstance(s, list) else []

def calculate_metrics(evidence_pages, retrieved_pages, k_values=[1, 3, 5, 10]):
    """
    计算检索评估指标
    
    Args:
        evidence_pages: 真实证据页面列表 [2, 22, 23]
        retrieved_pages: 检索返回的页面排名 [22, 3, 23, 6, 2, ...]
        k_values: 计算Hit Rate的K值列表
    
    Returns:
        dict: 包含各项指标的字典
    """
    evidence_set = set(evidence_pages)
    retrieved_list = retrieved_pages[:max(k_values)] if len(retrieved_pages) >= max(k_values) else retrieved_pages
    
    # 1. Hit Rate @ K: 是否在Top-K中命中最少一个证据页
    hit_rates = {}
    for k in k_values:
        top_k_retrieved = set(retrieved_list[:k])
        hit_rates[f'hit_rate@{k}'] = 1.0 if len(evidence_set & top_k_retrieved) > 0 else 0.0
    
    # 2. Recall @ K: 在Top-K中命中的证据页比例
    recalls = {}
    for k in k_values:
        top_k_retrieved = set(retrieved_list[:k])
        if len(evidence_set) > 0:
            recalls[f'recall@{k}'] = len(evidence_set & top_k_retrieved) / len(evidence_set)
        else:
            recalls[f'recall@{k}'] = 0.0
    
    # 3. Mean Reciprocal Rank (MRR): 第一个命中位置的倒数
    mrr = 0.0
    for i, page in enumerate(retrieved_list):
        if page in evidence_set:
            mrr = 1.0 / (i + 1)
            break
    
    # 4. Mean Average Precision (MAP): 所有证据页的平均精确率
    ap = 0.0
    num_hits = 0
    for i, page in enumerate(retrieved_list):
        if page in evidence_set:
            num_hits += 1
            ap += num_hits / (i + 1)
    if len(evidence_set) > 0:
        ap = ap / len(evidence_set)
    else:
        ap = 0.0
    
    # 5. 第一个证据页的位置
    first_evidence_pos = None
    for i, page in enumerate(retrieved_list):
        if page in evidence_set:
            first_evidence_pos = i + 1
            break
    
    # 6. 所有证据页的平均位置
    avg_evidence_pos = None
    positions = []
    for i, page in enumerate(retrieved_list):
        if page in evidence_set:
            positions.append(i + 1)
    if positions:
        avg_evidence_pos = sum(positions) / len(positions)
    
    return {
        **hit_rates,
        **recalls,
        'mrr': mrr,
        'map': ap,
        'first_evidence_pos': first_evidence_pos,
        'avg_evidence_pos': avg_evidence_pos,
        'num_evidence_pages': len(evidence_set)
    }

def main():
    # 加载数据
    with open('/gz-data/analyze/matched_failed_samples.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"总共加载了 {len(samples)} 个错误样本\n")
    
    # 计算Base方法的指标
    base_metrics = []
    # 计算Beamsearch方法的指标
    beam_metrics = []
    
    # 统计各种情况
    base_hit_counts = defaultdict(int)
    beam_hit_counts = defaultdict(int)
    
    for sample in samples:
        evidence_pages = parse_list_string(sample.get('evidence_pages', []))
        
        # Base方法
        base_ranking = parse_list_string(sample.get('base_pages_ranking_', []))
        base_result = calculate_metrics(evidence_pages, base_ranking)
        base_metrics.append(base_result)
        
        # 记录命中情况
        if base_result['hit_rate@1']: base_hit_counts['hit@1'] += 1
        if base_result['hit_rate@3']: base_hit_counts['hit@3'] += 1
        if base_result['hit_rate@5']: base_hit_counts['hit@5'] += 1
        if base_result['hit_rate@10']: base_hit_counts['hit@10'] += 1
        
        # Beamsearch方法
        beam_ranking = parse_list_string(sample.get('pages_ranking', []))
        beam_result = calculate_metrics(evidence_pages, beam_ranking)
        beam_metrics.append(beam_result)
        
        # 记录命中情况
        if beam_result['hit_rate@1']: beam_hit_counts['hit@1'] += 1
        if beam_result['hit_rate@3']: beam_hit_counts['hit@3'] += 1
        if beam_result['hit_rate@5']: beam_hit_counts['hit@5'] += 1
        if beam_result['hit_rate@10']: beam_hit_counts['hit@10'] += 1
    
    # 汇总统计
    print("=" * 70)
    print("错误样本检索性能分析报告")
    print("=" * 70)
    
    print("\n【1. Hit Rate 命中率分析】")
    print("-" * 50)
    print(f"{'指标':<15} {'Base方法':>15} {'Beamsearch':>15} {'提升':>15}")
    print("-" * 50)
    
    for k in [1, 3, 5, 10]:
        base_hr = base_hit_counts[f'hit@{k}'] / len(samples) * 100
        beam_hr = beam_hit_counts[f'hit@{k}'] / len(samples) * 100
        improvement = beam_hr - base_hr
        print(f"Hit Rate @{k:<2} {base_hr:>14.1f}% {beam_hr:>14.1f}% {improvement:>+14.1f}%")
    
    print("\n【2. 检索质量指标汇总】")
    print("-" * 50)
    
    # 计算平均MRR和MAP
    base_mrr = np.mean([m['mrr'] for m in base_metrics])
    beam_mrr = np.mean([m['mrr'] for m in beam_metrics])
    base_map = np.mean([m['map'] for m in base_metrics])
    beam_map = np.mean([m['map'] for m in beam_metrics])
    
    # 计算平均位置
    base_first_pos = np.mean([m['first_evidence_pos'] for m in base_metrics if m['first_evidence_pos']])
    beam_first_pos = np.mean([m['first_evidence_pos'] for m in beam_metrics if m['first_evidence_pos']])
    
    base_avg_pos = np.mean([m['avg_evidence_pos'] for m in base_metrics if m['avg_evidence_pos']])
    beam_avg_pos = np.mean([m['avg_evidence_pos'] for m in beam_metrics if m['avg_evidence_pos']])
    
    print(f"{'指标':<25} {'Base方法':>15} {'Beamsearch':>15}")
    print("-" * 50)
    print(f"{'MRR (Mean Reciprocal Rank)':<25} {base_mrr:>15.4f} {beam_mrr:>15.4f}")
    print(f"{'MAP (Mean Average Precision)':<25} {base_map:>15.4f} {beam_map:>15.4f}")
    print(f"{'第一个证据页平均位置':<25} {base_first_pos:>15.1f} {beam_first_pos:>15.1f}")
    print(f"{'所有证据页平均位置':<25} {base_avg_pos:>15.1f} {beam_avg_pos:>15.1f}")
    
    print("\n【3. 检索失败模式分析】")
    print("-" * 50)
    
    # 分析Base方法命中但Beamsearch未命中的情况
    base_only_hit = 0
    beam_only_hit = 0
    both_hit = 0
    neither_hit = 0
    
    for i, sample in enumerate(samples):
        evidence_pages = parse_list_string(sample.get('evidence_pages', []))
        base_ranking = parse_list_string(sample.get('base_pages_ranking_', []))[:10]
        beam_ranking = parse_list_string(sample.get('pages_ranking', []))[:10]
        
        base_hit = len(set(evidence_pages) & set(base_ranking)) > 0
        beam_hit = len(set(evidence_pages) & set(beam_ranking)) > 0
        
        if base_hit and beam_hit:
            both_hit += 1
        elif base_hit and not beam_hit:
            base_only_hit += 1
        elif not base_hit and beam_hit:
            beam_only_hit += 1
        else:
            neither_hit += 1
    
    print(f"两种方法都命中:     {both_hit:>5} ({both_hit/len(samples)*100:.1f}%)")
    print(f"仅Base命中:         {base_only_hit:>5} ({base_only_hit/len(samples)*100:.1f}%)")
    print(f"仅Beamsearch命中:   {beam_only_hit:>5} ({beam_only_hit/len(samples)*100:.1f}%)")
    print(f"两种方法都未命中:   {neither_hit:>5} ({neither_hit/len(samples)*100:.1f}%)")
    
    print("\n【4. 关键发现】")
    print("-" * 50)
    
    # 计算提升比例
    if base_only_hit > 0:
        print(f"• Beamsearch相对于Base的改进样本数: {beam_only_hit - base_only_hit}")
    else:
        print(f"• Beamsearch相对于Base的改进样本数: {beam_only_hit}")
    
    # 统计证据页数量分布
    evidence_counts = [len(parse_list_string(s.get('evidence_pages', []))) for s in samples]
    print(f"\n• 错误样本的证据页数量分布:")
    print(f"  - 平均需要 {np.mean(evidence_counts):.1f} 个证据页")
    print(f"  - 最多需要 {max(evidence_counts)} 个证据页")
    print(f"  - 只需1个证据页的样本: {evidence_counts.count(1)} 个 ({evidence_counts.count(1)/len(samples)*100:.1f}%)")
    print(f"  - 需要2-3个证据页的样本: {sum(1 for c in evidence_counts if 2 <= c <= 3)} 个")
    print(f"  - 需要4+个证据页的样本: {sum(1 for c in evidence_counts if c >= 4)} 个")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
