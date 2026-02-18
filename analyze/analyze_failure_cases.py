"""
深入分析失败案例
找出Base和Beamsearch都无法命中的样本特征
"""

import json
import ast

def parse_list_string(s):
    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except:
            return []
    return s if isinstance(s, list) else []

def main():
    with open('/gz-data/analyze/matched_failed_samples.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print("=" * 80)
    print("深入分析：检索阶段失败的样本特征")
    print("=" * 80)
    
    # 找出两种方法都未命中的样本
    failed_retrieval = []
    base_only_failed = []
    beam_only_failed = []
    
    for sample in samples:
        evidence_pages = parse_list_string(sample.get('evidence_pages', []))
        base_ranking = parse_list_string(sample.get('base_pages_ranking_', []))[:10]
        beam_ranking = parse_list_string(sample.get('pages_ranking', []))[:10]
        
        base_hit = len(set(evidence_pages) & set(base_ranking)) > 0
        beam_hit = len(set(evidence_pages) & set(beam_ranking)) > 0
        
        if not base_hit and not beam_hit:
            failed_retrieval.append(sample)
        elif base_hit and not beam_hit:
            beam_only_failed.append(sample)
        elif not base_hit and beam_hit:
            base_only_failed.append(sample)
    
    print(f"\n【检索完全失败的样本数】: {len(failed_retrieval)} ({len(failed_retrieval)/len(samples)*100:.1f}%)")
    
    # 分析失败样本的特征
    print("\n--- 失败样本的证据页数量分布 ---")
    evidence_counts = [len(parse_list_string(s.get('evidence_pages', []))) for s in failed_retrieval]
    unique_counts = sorted(set(evidence_counts))
    for count in unique_counts:
        freq = evidence_counts.count(count)
        print(f"需要 {count:2d} 个证据页的样本: {freq:3d} 个 ({freq/len(failed_retrieval)*100:.1f}%)")
    
    print("\n--- 失败样本的问题类型分布 ---")
    answer_formats = {}
    for sample in failed_retrieval:
        fmt = sample.get('answer_format', 'Unknown')
        answer_formats[fmt] = answer_formats.get(fmt, 0) + 1
    
    for fmt, count in sorted(answer_formats.items(), key=lambda x: -x[1]):
        print(f"{fmt:20s}: {count:3d} 个 ({count/len(failed_retrieval)*100:.1f}%)")
    
    print("\n--- 失败样本的文档类型分布 ---")
    doc_types = {}
    for sample in failed_retrieval:
        dtype = sample.get('doc_type', 'Unknown')
        doc_types[dtype] = doc_types.get(dtype, 0) + 1
    
    for dtype, count in sorted(doc_types.items(), key=lambda x: -x[1])[:10]:
        print(f"{dtype:40s}: {count:3d} 个")
    
    # 展示一些典型失败案例
    print("\n" + "=" * 80)
    print("典型失败案例示例 (前5个)")
    print("=" * 80)
    
    for i, sample in enumerate(failed_retrieval[:5]):
        print(f"\n案例 {i+1}:")
        print(f"  ID: {sample['id']}")
        print(f"  问题: {sample['question']}")
        print(f"  正确答案: {sample['answer']}")
        print(f"  预测答案: {sample.get('pred_ans', 'N/A')}")
        print(f"  证据页(真实): {sample['evidence_pages']}")
        print(f"  Base检索Top10: {parse_list_string(sample.get('base_pages_ranking_', []))[:10]}")
        print(f"  Beam检索Top10: {parse_list_string(sample.get('pages_ranking', []))[:10]}")
    
    # 分析Beamsearch反而退化的案例
    print("\n" + "=" * 80)
    print("Beamsearch退化的案例 (Base命中但Beamsearch未命中)")
    print("=" * 80)
    
    if beam_only_failed:
        print(f"共有 {len(beam_only_failed)} 个这类案例")
        for i, sample in enumerate(beam_only_failed[:3]):
            print(f"\n案例 {i+1}:")
            print(f"  ID: {sample['id']}")
            print(f"  问题: {sample['question']}")
            print(f"  证据页(真实): {sample['evidence_pages']}")
            print(f"  Base检索Top10: {parse_list_string(sample.get('base_pages_ranking_', []))[:10]}")
            print(f"  Beam检索Top10: {parse_list_string(sample.get('pages_ranking', []))[:10]}")
    else:
        print("没有发现Beamsearch退化的案例")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
