"""
Error Analysis Script for MoLoRAG Retrieval Results

Classification Rules:
1. Initialization Failure (锚点丢失):
   - GT not in Initial Top-w AND GT not in the 1-hop neighborhood of Initial Top-w.
   - Interpretation: s^{sem} failed completely.

2. Graph Disconnect (断路):
   - GT is in the 1-hop neighborhood of Initial Top-w, but was NOT visited.
   - Interpretation: s^{sem} worked (found neighbors), but traversal missed it (edge score < theta).

3. Scoring/Ranking Failure (误杀):
   - GT was visited, but final score ranked outside Top-K.
   - Interpretation: VLM scoring or final ranking is insufficient.
"""

import json
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple

def load_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_pages(pages_str: str) -> List[int]:
    """Safely parse string representation of lists like '[1, 2, 3]'"""
    if isinstance(pages_str, str):
        pages_str = pages_str.strip()
        if pages_str.startswith('[') and pages_str.endswith(']'):
            try:
                return eval(pages_str)
            except:
                return []
    elif isinstance(pages_str, list):
        return pages_str
    return []

def analyze_errors(samples: List[Dict], top_k: int = 5, initial_w: int = 3) -> Dict[str, Any]:
    """
    Analyzes errors in the retrieval results.
    
    Note: To distinguish "Initialization Failure" from "Graph Disconnect", 
    we ideally need the Graph Topology. Since we don't have it, we will infer
    "Potential Connectivity" based on the assumption that pages not in the 
    initial beam but visited in the full list are likely "1-hop neighbors".
    
    However, without explicit graph data, we can only differentiate based on
    whether the GT was VISITED or not, and whether it was in the INITIAL BEAM.
    
    Strategy:
    1. If GT not in Initial Beam AND not visited -> Initialization Failure.
    2. If GT not in Initial Beam BUT visited -> Potential Graph Disconnect (Inferred).
    3. If GT in Initial Beam BUT ranked > K -> Scoring Failure.
    """
    
    stats = {
        "total_samples": len(samples),
        "success": 0,
        "failure": 0,
        "by_type": {
            "Scoring/Ranking Failure (误杀)": 0,
            "Initialization Failure (锚点丢失)": 0,
            "Potential Graph Disconnect (潜在断路)": 0, # Inferred from logic
            "Unknown/Logic Error": 0
        },
        "details": []
    }

    for sample in samples:
        doc_id = sample.get("doc_id", "Unknown")
        q_id = sample.get("id", "Unknown")
        
        # Parse GT
        gt_pages = set(parse_pages(sample.get("evidence_pages", "[]")))
        
        # Parse Retrieved Pages (Full List)
        retrieved_pages = parse_pages(sample.get("pages_ranking", "[]"))
        
        if not retrieved_pages:
            stats["failure"] += 1
            stats["by_type"]["Initialization Failure (锚点丢失)"] += 1
            continue

        retrieved_set = set(retrieved_pages)
        
        # 1. Check visited status
        visited_gt_pages = gt_pages.intersection(retrieved_set)
        
        # 2. Check Initial Top-w status
        initial_beam = set(retrieved_pages[:initial_w])
        gt_in_initial_beam = bool(gt_pages.intersection(initial_beam))

        # 3. Classification Logic
        
        # Case A: Success
        found_in_top_k = any(p in gt_pages for p in retrieved_pages[:top_k])
        if found_in_top_k:
            stats["success"] += 1
            continue
            
        # Case B: Error Analysis
        
        if visited_gt_pages:
            if not gt_in_initial_beam:
                # Logic: GT was NOT in Initial Beam, but WAS visited later.
                # In MoLoRAG, visited nodes come from neighbors.
                # If it wasn't in the beam, it must have been reached via an edge.
                # This implies s^{sem} found neighbors, but they didn't make the Beam cut.
                # This is "Graph Disconnect" or "Traversal Failure" (Beam search logic).
                stats["failure"] += 1
                stats["by_type"]["Potential Graph Disconnect (潜在断路)"] += 1
                stats["details"].append({
                    "doc_id": doc_id,
                    "q_id": q_id,
                    "error_type": "Potential Graph Disconnect (潜在断路)",
                    "reason": "GT visited but not in Initial Top-w. Likely missed in traversal.",
                    "gt_pages": list(gt_pages),
                    "initial_beam": list(retrieved_pages[:initial_w])
                })
            else:
                # Logic: GT was in Initial Beam, but rank > K.
                # This is purely a scoring/ranking failure.
                stats["failure"] += 1
                stats["by_type"]["Scoring/Ranking Failure (误杀)"] += 1
                rank = next((i+1 for i, p in enumerate(retrieved_pages) if p in gt_pages), -1)
                stats["details"].append({
                    "doc_id": doc_id,
                    "q_id": q_id,
                    "error_type": "Scoring/Ranking Failure (误杀)",
                    "reason": f"GT in Initial Top-w but ranked at {rank} (>={top_k}).",
                    "gt_pages": list(gt_pages)
                })
                
        else: # GT not visited at all
            if not gt_in_initial_beam:
                # Logic: GT was not in Initial Beam AND not visited.
                # Since it wasn't in the beam, it couldn't be a starting point for traversal.
                # Therefore, it wasn't reached.
                # If we assume the full 'retrieved_pages' list represents all reachable nodes,
                # then GT is disconnected from the start.
                stats["failure"] += 1
                stats["by_type"]["Initialization Failure (锚点丢失)"] += 1
                stats["details"].append({
                    "doc_id": doc_id,
                    "q_id": q_id,
                    "error_type": "Initialization Failure (锚点丢失)",
                    "reason": "GT not in Initial Top-w and not visited. Semantic Search failed.",
                    "gt_pages": list(gt_pages)
                })
            else:
                # Logic Error: Should not happen
                stats["failure"] += 1
                stats["by_type"]["Unknown/Logic Error"] += 1
                stats["details"].append({
                    "doc_id": doc_id,
                    "q_id": q_id,
                    "error_type": "Unknown/Logic Error",
                    "reason": "GT in Initial Top-w but not visited. Algorithm logic error.",
                    "gt_pages": list(gt_pages)
                })

    return stats

def print_summary(stats: Dict):
    print("\n" + "="*60)
    print("MoLoRAG Error Analysis Summary")
    print("="*60)
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Success (In Top-K): {stats['success']}")
    print(f"Failures: {stats['failure']}")
    
    print("\nFailure Breakdown:")
    total = stats['total_samples']
    for name, count in stats["by_type"].items():
        if count > 0:
            print(f"  - {name}: {count} ({count/total*100:.2f}%)")
    
    print("\n" + "-"*60)
    print("Insights:")
    init_fail = stats["by_type"]["Initialization Failure (锚点丢失)"]
    graph_fail = stats["by_type"]["Potential Graph Disconnect (潜在断路)"]
    score_fail = stats["by_type"]["Scoring/Ranking Failure (误杀)"]

    if init_fail > 0:
        print(f"⚠️  Semantic Search Issues: {init_fail} samples failed at initialization (s^{{sem}} failed).")
    if graph_fail > 0:
        print(f"⚠️  Traversal/Beam Issues: {graph_fail} samples missed during graph walk (s^{{logi}} or threshold issue).")
    if score_fail > 0:
        print(f"⚠️  Final Ranking Issues: {score_fail} samples failed due to low relevance score (s failed).")

def main():
    parser = argparse.ArgumentParser(description="Analyze MoLoRAG Retrieval Errors")
    parser.add_argument("--input_file", type=str, 
                        default="/gz-data/dataset/retrieved/samples_MMLong_beamsearch_LoRA.json",
                        help="Path to the JSON file with retrieval results")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K threshold (default: 5)")
    parser.add_argument("--initial_w", type=int, default=3, help="Initial Top-w size (default: 3)")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input_file}...")
    samples = load_data(args.input_file)
    print(f"Loaded {len(samples)} samples.")
    
    stats = analyze_errors(samples, top_k=args.top_k, initial_w=args.initial_w)
    print_summary(stats)
    
    # Save report
    output_json = args.input_file.replace(".json", "_analysis.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {output_json}")

if __name__ == "__main__":
    main()
