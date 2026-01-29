import torch 
from tqdm import tqdm 
from colpali_engine.models import ColPali, ColPaliProcessor
import argparse
import sys 
sys.path.append("../")
from utils import load_all_doc_embeddings, construct_page_graph
import json 
from pdf2image import convert_from_path
import os 
import re
import numpy as np
from collections import defaultdict
from copy import deepcopy
from data_collection import generate_relevance_prompt as rel_prompt, generate_relevance_prompt_detailed as advanced_rel_prompt

# BM25 相关导入
from utils.datautil import (
    load_or_build_bm25_index,
    normalize_scores,
    select_diverse_top_k
)


def query_vlm_relevance(query, doc_info, vlm_model):
    doc_id, page_num = doc_info 
    if not os.path.exists(f"/gz-data/tmp/tmp_imgs/{args.dataset}/{doc_id}-{page_num}.png"):
        doc_snapshot = convert_from_path(pdf_path=f"/gz-data/dataset/{args.dataset}/{doc_id}.pdf", first_page=page_num, last_page=page_num, dpi=144)[0] # resolution is fixed to 144
        doc_snapshot.save(f"/gz-data/tmp/tmp_imgs/{args.dataset}/{doc_id}-{page_num}.png", "PNG")
    
    img_path = f"/gz-data/tmp/tmp_imgs/{args.dataset}/{doc_id}-{page_num}.png"
    prompt = rel_prompt(query) if args.dataset == "MMLong" else advanced_rel_prompt(query)
    
    try:
        response = get_response_concat(vlm_model, prompt, img_path, max_new_tokens=16, temperature=1)
        score_match = re.search(r'[1-5]', response)
        if score_match:
            relevance_score = int(score_match.group(0))
        else:
            relevance_score = 3
    except Exception as e:
        print(f"Error in VLM for {doc_id}-{page_num}; Exception {e}")
        relevance_score = 1
    
    return relevance_score
    

class DocumentRetriever:
    """
    A unified class for document retrieval with multiple retrieval strategies
    Supports both dense (ColPali) and sparse (BM25) retrieval, with hybrid options
    """
    def __init__(self, encoder: ColPali, processor: ColPaliProcessor, device: torch.device, batch_size=512):
        self.encoder = encoder 
        self.processor = processor 
        self.device = device 
        self.batch_size = batch_size
        # BM25 索引缓存 {doc_id: (bm25_index, page_texts)}
        self.bm25_cache = {}

    def compute_scores(self, query, all_embeds):
        queries = self.processor.process_queries(queries=[query]).to(self.device)
        query_embeds = self.encoder(**queries)

        all_scores = []
        for idx in range(0, all_embeds.shape[0], self.batch_size):
            batch_embeds = all_embeds[idx: idx+self.batch_size]
            batch_embeds = torch.FloatTensor(batch_embeds).to(device=self.device, dtype=query_embeds.dtype)

            with torch.no_grad():
                tmp_scores = self.processor.score_multi_vector(query_embeds, batch_embeds)
                if len(tmp_scores.shape) > 1:
                    tmp_scores = tmp_scores[0]

            all_scores.append(tmp_scores)
        scores = torch.cat(all_scores, dim=0).cpu()
        del all_scores, queries, query_embeds 
        return scores
    
    def get_bm25_scores(self, query, doc_id, dataset_name):
        """
        获取查询与文档的 BM25 分数

        参数:
            query: 查询文本
            doc_id: 文档 ID
            dataset_name: 数据集名称

        返回:
            numpy.ndarray: 每页的 BM25 分数 (归一化到 [0,1])
        """
        # 加载或构建 BM25 索引
        if doc_id not in self.bm25_cache:
            bm25_index, page_texts = load_or_build_bm25_index(doc_id, dataset_name)
            self.bm25_cache[doc_id] = (bm25_index, page_texts)
        else:
            bm25_index, page_texts = self.bm25_cache[doc_id]

        if bm25_index is None:
            # 所有页面为空，返回零分数
            n_pages = len(page_texts)
            return np.zeros(n_pages)

        # 计算 BM25 分数
        import re
        def simple_tokenizer(text):
            text = text.lower()
            tokens = re.findall(r'\w+', text)
            return tokens

        tokenized_query = simple_tokenizer(query)
        raw_scores = bm25_index.get_scores(tokenized_query)

        # 处理空文档：创建完整页面数的分数数组
        full_scores = np.zeros(len(page_texts))
        valid_idx = 0
        for i, text in enumerate(page_texts):
            if text.strip():
                full_scores[i] = raw_scores[valid_idx]
                valid_idx += 1

        # 归一化到 [0, 1]
        normalized_scores = normalize_scores(full_scores, 0.0, 1.0)

        return normalized_scores

    def compute_hybrid_scores(self, query, all_embeds, doc_id, dataset_name,
                               dense_weight=0.6, sparse_weight=0.4):
        """
        计算混合检索分数

        参数:
            query: 查询文本
            all_embeds: 文档嵌入向量
            doc_id: 文档 ID
            dataset_name: 数据集名称
            dense_weight: 稠密检索权重
            sparse_weight: 稀疏检索权重

        返回:
            tuple: (混合分数, 稠密分数, 稀疏分数)
        """
        # 1. 计算稠密检索分数
        dense_scores = self.compute_scores(query, all_embeds)
        # 归一化到 [0, 1]
        dense_scores_norm = normalize_scores(dense_scores.numpy(), 0.0, 1.0)

        # 2. 计算稀疏检索分数 (BM25)
        sparse_scores = self.get_bm25_scores(query, doc_id, dataset_name)

        # 3. 确保长度一致
        min_len = min(len(dense_scores_norm), len(sparse_scores))
        dense_scores_norm = dense_scores_norm[:min_len]
        sparse_scores = sparse_scores[:min_len]

        # 4. 混合分数
        hybrid_scores = dense_weight * dense_scores_norm + sparse_weight * sparse_scores

        return hybrid_scores, dense_scores_norm, sparse_scores
    
    def base_retrieve(self, query, all_embeds, top_k=10):
        scores = self.compute_scores(query, all_embeds)
        top_indices = scores.argsort(dim=-1, descending=True)[:top_k].tolist()
        top_scores = scores[top_indices].tolist()
        # 1-indexed
        return [idx+1 for idx in top_indices], top_scores
    
    def vlm_retrieve(self, query, all_embeds, graph, doc_id, dataset_name,
                      beam_width=3, max_hop=5, verbose=True,
                      use_hybrid=False, dense_weight=0.6, sparse_weight=0.4,
                      candidate_multiplier=2):
        """
        VLM增强的Beamsearch检索方法，支持混合稠密+稀疏检索

        参数:
            query: 用户查询
            all_embeds: 文档所有页面的嵌入向量
            graph: 页面邻接图 {page_idx: [neighbor_idx, ...]}
            doc_id: 文档ID
            dataset_name: 数据集名称 (用于 BM25 检索)
            beam_width: 每轮保留的候选数量
            max_hop: 最大搜索跳数
            verbose: 是否打印详细信息
            use_hybrid: 是否使用混合检索 (BM25 + ColPali)
            dense_weight: 稠密检索权重
            sparse_weight: 稀疏检索权重
            candidate_multiplier: 初始候选池扩大倍数
        """
        n_pages = all_embeds.shape[0]

        # 计算初始分数
        if use_hybrid:
            hybrid_scores, dense_scores, sparse_scores = self.compute_hybrid_scores(
                query, all_embeds, doc_id, dataset_name,
                dense_weight=dense_weight, sparse_weight=sparse_weight
            )

            if verbose:
                print(f"Dense scores (top5): {dense_scores.argsort()[::-1][:5] + 1}")
                print(f"Sparse scores (top5): {sparse_scores.argsort()[::-1][:5] + 1}")
                print(f"Hybrid scores (top5): {hybrid_scores.argsort()[::-1][:5] + 1}")

            # 使用混合分数作为基础分数
            scores = torch.tensor(hybrid_scores)
        else:
            scores = self.compute_scores(query, all_embeds)

        # 归一化分数用于后续计算
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()
        score_range = max_score - min_score if max_score > min_score else 1.0

        vlm_score_cache, vlm_query_times = {}, 0
        score_dict = {i: (scores[i].item() - min_score) / score_range for i in range(n_pages)}

        # 扩大初始候选池，然后使用多样性选择
        initial_pool_size = beam_width * candidate_multiplier
        initial_pool_size = min(initial_pool_size, n_pages)  # 不超过总页数

        if use_hybrid:
            # 使用混合分数排序
            initial_pool = hybrid_scores.argsort()[::-1][:initial_pool_size].tolist()
        else:
            # 使用原始稠密分数排序
            initial_pool = scores.argsort(dim=-1, descending=True)[:initial_pool_size].tolist()

        # 使用多样性选择策略选取 beam_width 个初始候选
        initial_scores_for_diversity = [hybrid_scores[i] if use_hybrid else scores[i].item()
                                        for i in initial_pool]
        initial_beam = select_diverse_top_k(initial_pool, initial_scores_for_diversity, beam_width)

        visited = set(initial_beam)

        # VLM 评估初始 beam 中的每个页面
        for node in initial_beam:
            vlm_score = query_vlm_relevance(query, (doc_id, node+1), vlm_model)
            vlm_query_times += 1
            vlm_score_cache[node] = vlm_score
            norm_vlm_score = (vlm_score - 1.0) / 4.0

            combined_score = args.alpha * score_dict[node] + (1.0 - args.alpha) * norm_vlm_score
            score_dict[node] = combined_score

        if verbose:
            print(f"Initial Pool: {[node_id+1 for node_id in initial_pool]}")
            print(f"Initial Beam (after diversity selection): {[node_id+1 for node_id in initial_beam]}")
            print(f"Initial Beam Scores: {[round(score_dict[node], 3) for node in initial_beam]}")

        result_dict = {node: score_dict[node] for node in initial_beam}

        # Start the beam search
        current_beam = initial_beam
        for hop in range(max_hop):
            candidates = []
            for node in current_beam:
                candidate_neighbors = graph.get(node, [])
                for neighbor in candidate_neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)

                        sim_score = score_dict[neighbor]
                        vlm_score = query_vlm_relevance(query, (doc_id, neighbor+1), vlm_model)
                        vlm_query_times += 1
                        vlm_score_cache[neighbor] = vlm_score
                        norm_vlm_score = (vlm_score - 1.0) / 4.0

                        combined_score = args.alpha * sim_score + (1.0 - args.alpha) * norm_vlm_score
                        score_dict[neighbor] = combined_score
                        candidates.append((neighbor, combined_score))
                        result_dict[neighbor] = combined_score

            if not candidates:
                break

            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            current_beam = [node for node, _ in candidates]
            if verbose:
                print(f"Hop {hop+1}: Beam = {[node_id+1 for node_id in current_beam]}; Scores = {[round(score_dict[node], 3) for node in current_beam]}")
        
        final_results = [(node, score) for node, score in result_dict.items() if score >= args.threshold]
        final_results = sorted(final_results, key=lambda x: x[1], reverse=True)
        evidence_pages = [node+1 for node, _ in final_results]
        page_scores = [score for _, score in final_results]
        torch.cuda.empty_cache()

        print(f"Total Pages {all_embeds.shape[0]}; VLM Query Times: {vlm_query_times}")
        return evidence_pages, page_scores


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="MMLong", choices=["MMLong", "LongDocURL", "FetaTab", "PaperTab"])
    args.add_argument("--method", type=str, default="base", choices=["base", "beamsearch"])
    args.add_argument("--emb_root", type=str, default="/gz-data/tmp/tmp_embs")
    args.add_argument("--device", type=str, default="cuda:0")
    args.add_argument("--encoder", type=str, default="vidore/colpali")
    args.add_argument("--top_k", type=int, default=20)
    args.add_argument("--model_name", type=str, default="QwenVL-3B", choices=["QwenVL-3B", "QwenVL-3B-lora"])
    args.add_argument("--threshold", type=float, default=0.3) # 0.3 default; 0.4 for MMLong + QwenVL-3B-lora

    # beamsearch specific
    args.add_argument("--alpha", type=float, default=0.4) # 0.4 for MMLong and (currently) 0.6 for LongDocURL
    args.add_argument("--sim_measure", type=str, default="cosine")
    args.add_argument("--beam_width", type=int, default=3)
    args.add_argument("--max_hop", type=int, default=4)
    args.add_argument("--beam_verbose", type=bool, default=True)
    
    # Hybrid BM25 + Dense retrieval options
    args.add_argument("--use_hybrid", action="store_true", help="Use hybrid retrieval (BM25 + ColPali)")
    args.add_argument("--dense_weight", type=float, default=0.6, help="Weight for dense (ColPali) scores in hybrid retrieval")
    args.add_argument("--sparse_weight", type=float, default=0.4, help="Weight for sparse (BM25) scores in hybrid retrieval")
    args.add_argument("--candidate_multiplier", type=int, default=2, help="Multiplier for initial candidate pool size")
    args = args.parse_args()

    device = torch.device(args.device)
    doc2emb = load_all_doc_embeddings(f"{args.emb_root}/{args.dataset}") # dict format {doc_id: doc_emb}
    model, vlm_model = ColPali.from_pretrained(args.encoder, torch_dtype=torch.bfloat16, device_map=device).eval(), None
    processor = ColPaliProcessor.from_pretrained(args.encoder)
    doc_retriever = DocumentRetriever(encoder=model, processor=processor, device=device)
    
    samples = json.load(open(f"/gz-data/dataset/samples_{args.dataset}.json", 'r'))
    vlm_suffix = "_LoRA" if args.method == "beamsearch" and "lora" in args.model_name else ""
    retrieve_file = f"/gz-data/dataset/retrieved/samples_{args.dataset}_{args.method}{vlm_suffix}.json"

    if args.method == "beamsearch":
        from VLMModels.Qwen_VL import init_model, get_response_concat
        vlm_model = init_model(args.model_name, device=device)
        doc2graph = {}
        for doc_id, doc_emb in tqdm(doc2emb.items(), desc="Constructing Page Graph"):
            cur_graph = construct_page_graph(doc_emb, threshold=0.8, sim_measure=args.sim_measure)
            doc2graph[doc_id] = deepcopy(cur_graph)

    for sample in tqdm(samples, desc="Retrieving"):
        query = sample["question"]
        target_doc = sample["doc_id"].replace(".pdf", "")
        target_doc_embedding = doc2emb[target_doc]
        if args.method == "base":
            ranked_pages, page_scores = doc_retriever.base_retrieve(query, target_doc_embedding, top_k=args.top_k)
        elif args.method == "beamsearch":
            target_graph = doc2graph.get(target_doc, defaultdict(list))
            try:
                assert target_graph is not None
            except Exception as e: 
                print(f'Error in graph', target_graph)
                target_graph = defaultdict(list)
                
            ranked_pages, page_scores = doc_retriever.vlm_retrieve(
                query, target_doc_embedding, target_graph, target_doc,
                dataset_name=args.dataset,
                beam_width=args.beam_width, max_hop=args.max_hop, verbose=args.beam_verbose,
                use_hybrid=args.use_hybrid, dense_weight=args.dense_weight,
                sparse_weight=args.sparse_weight, candidate_multiplier=args.candidate_multiplier
            )
        
        sample["pages_ranking"] = str(ranked_pages)
        sample["pages_scores"] = str(page_scores)
        if "evidence_pages" in sample:
            print("Ground-truth", sample["evidence_pages"])
        print("Prediction", ranked_pages[:5], "\n")
        
        os.makedirs(os.path.dirname(retrieve_file), exist_ok=True)
        json.dump(samples, open(retrieve_file, 'w'), indent=4)
