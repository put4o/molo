import os 
from tqdm import tqdm 
import torch 
from copy import deepcopy
import numpy as np
from collections import defaultdict
from .general import compute_embed_similarity
import math 
from PIL import Image


def prepare_files(root_dir: str, suffix=".pdf"):
    """Prepare the list of files in the `root_dir` with the specified `suffix`"""
    target_files = [file for file in os.listdir(root_dir) if file.endswith(suffix)]
    return target_files


def load_all_doc_embeddings(root_dir):
    """Load all embeddings and organize them."""
    emb_files = sorted(prepare_files(root_dir, suffix=".pt"))
    docid2emb = {}

    for emb_file in tqdm(emb_files, desc="Loading and Organizing Embeddings"):
        embeds = torch.load(f"{root_dir}/{emb_file}", map_location="cpu", weights_only=True).detach().numpy()
        
        doc_id = emb_file.replace(".pt", "")
        docid2emb[doc_id] = embeds

    return docid2emb


def construct_page_graph(doc_emb, threshold=0.7, k_value=5, sim_measure="cosine"):
    """Construct a page graph based on the similarity between page embeddings."""
    n_pages, _, _ = doc_emb.shape
    if n_pages <= 3: # skip small documents
        return None

    edges = []
    
    sim_matrix = np.zeros((n_pages, n_pages))
    for i in range(n_pages):
        for j in range(i+1, n_pages):
            vec_i, vec_j = doc_emb[i], doc_emb[j]
            sim_score = compute_embed_similarity(vec_i, vec_j, sim_func=sim_measure)
            sim_matrix[i][j] = sim_score
            sim_matrix[j][i] = sim_score
    
    # k-NN Graph
    for i in range(n_pages):
        sim_scores = sim_matrix[i] 
        top_k_indices = np.argsort(sim_scores)[::-1][:k_value]

        for j in top_k_indices:
            if sim_scores[j] >= threshold:
                edges.append((i, j))
    
    page_graph_dict = defaultdict(list)
    for u, v in edges:
        page_graph_dict[int(u)].append(int(v))
        page_graph_dict[int(v)].append(int(u))
    
    page_graph_dict = {k: list(set(v)) for k, v in page_graph_dict.items()}
    # print(page_graph_dict, "\n")
    # print(f"Doc-{doc_id}-Graph # Nodes {n_pages}, # Edges {len(edges)}")
    return page_graph_dict


def convert_page_snapshot_to_image(doc_path, save_path, resolution=144, max_pages=1000):
    """Convert a PDF document to a list of images."""
    from pdf2image import convert_from_path
    page_snapshots = convert_from_path(doc_path, dpi=resolution)
    doc_id = doc_path.split("/")[-1].replace(".pdf", "")
    image_path_list = []
    for page_num, page_snapshot in enumerate(page_snapshots[:max_pages]):
        if not os.path.exists(f"{save_path}/{doc_id}-{page_num+1}.png"):
            page_snapshot.save(f"{save_path}/{doc_id}-{page_num+1}.png", "PNG")
        image_path_list.append(f"{save_path}/{doc_id}-{page_num+1}.png")
    
    return image_path_list


def concat_images(image_list, concat_num=1, column_num=3, name_suffix="concat"):
    """Concatenate a list of images into `concat_num` images."""
    interval = max(math.ceil(len(image_list) / concat_num), 1) # number of images in each batch
    concatenated_image_list = list()

    for i in range(0, len(image_list), interval):
        image_path = "-".join(image_list[0].split("-")[:-1]) + "-{}{}-{}.jpg".format(name_suffix, concat_num, i//interval)
        print(image_path)
        if not os.path.exists(image_path):
            images_this_batch = [
                Image.open(filename) for filename in image_list[i:i + interval]
            ]
            if column_num == 1:
                total_height = images_this_batch[0].height*len(images_this_batch)
            else:
                total_height = images_this_batch[0].height*((len(images_this_batch)-1)//column_num+1)

            concatenated_image = Image.new('RGB', (images_this_batch[0].width*column_num, total_height), 'white')
            x_offset, y_offset = 0, 0
            for cnt, image in enumerate(images_this_batch):
                concatenated_image.paste(image, (x_offset, y_offset))
                x_offset += image.width
                if (cnt+1)%column_num==0:
                    y_offset += image.height
                    x_offset = 0
            concatenated_image.save(image_path)
        concatenated_image_list.append(image_path)

    return concatenated_image_list


# ============================================================
# BM25 稀疏检索相关函数
# ============================================================

def extract_page_text_from_pdf(pdf_path, page_num=1):
    """
    从 PDF 中提取指定页面的文本

    参数:
        pdf_path: PDF 文件路径
        page_num: 页码 (从 1 开始)

    返回:
        str: 页面文本内容
    """
    import fitz  # pymupdf

    try:
        doc = fitz.open(pdf_path)
        # page_num 从 1 开始，需要转换为 0 索引
        page = doc[page_num - 1]
        text = page.get_text("text")
        doc.close()
        return text if text else ""
    except Exception as e:
        print(f"Error extracting text from {pdf_path} page {page_num}: {e}")
        return ""


def extract_all_pages_text(pdf_path, max_pages=None):
    """
    从 PDF 中提取所有页面的文本

    参数:
        pdf_path: PDF 文件路径
        max_pages: 最大提取页数 (None 表示全部)

    返回:
        list: 每页文本的列表
    """
    import fitz  # pymupdf

    page_texts = []
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc) if max_pages is None else min(len(doc), max_pages)

        for page_num in range(num_pages):
            page = doc[page_num]
            text = page.get_text("text")
            page_texts.append(text if text else "")

        doc.close()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")

    return page_texts


def build_bm25_index(doc_texts, tokenization="whitespace"):
    """
    构建 BM25 索引

    参数:
        doc_texts: 文档列表，每个元素是一页的文本
        tokenization: 分词方式 ("whitespace" 或 "nltk")

    返回:
        BM25Okapi: BM25 索引对象
    """
    from rank_bm25 import BM25Okapi
    import re

    def simple_tokenizer(text):
        """简单分词器：按空白字符分割，转小写"""
        text = text.lower()
        # 保留字母、数字，移除其他字符
        tokens = re.findall(r'\w+', text)
        return tokens

    def nltk_tokenizer(text):
        """使用 NLTK 分词"""
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        from nltk.tokenize import word_tokenize
        text = text.lower()
        return word_tokenize(text)

    if tokenization == "nltk":
        tokenizer = nltk_tokenizer
    else:
        tokenizer = simple_tokenizer

    # 对每个文档进行分词
    tokenized_docs = [tokenizer(text) for text in doc_texts]

    # 过滤空文档
    tokenized_docs = [doc for doc in tokenized_docs if len(doc) > 0]

    # 构建 BM25 索引
    bm25_index = BM25Okapi(tokenized_docs)

    return bm25_index


def bm25_retrieve(query, bm25_index, doc_texts=None, top_k=20):
    """
    使用 BM25 进行检索

    参数:
        query: 查询文本
        bm25_index: BM25 索引对象
        doc_texts: 原始文档文本列表 (用于过滤空文档)
        top_k: 返回的 top-k 结果数量

    返回:
        tuple: (页面索引列表, 分数列表) - 索引从 1 开始
    """
    import re

    def simple_tokenizer(text):
        """简单分词器"""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    tokenized_query = simple_tokenizer(query)

    # 计算查询与所有文档的 BM25 分数
    scores = bm25_index.get_scores(tokenized_query)

    # 如果有 doc_texts，过滤掉空文档的分数
    if doc_texts is not None:
        valid_mask = [len(text.strip()) > 0 for text in doc_texts]
        scores = scores[valid_mask]

    # 获取 top-k 结果
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_indices]

    # 返回 1 索引的页面编号
    return [idx + 1 for idx in top_indices], top_scores.tolist()


def load_or_build_bm25_index(doc_id, dataset_name, bm25_cache_dir="/gz-data/tmp/tmp_bm25"):
    """
    加载或构建文档的 BM25 索引

    参数:
        doc_id: 文档 ID (不含 .pdf 扩展名)
        dataset_name: 数据集名称
        bm25_cache_dir: BM25 索引缓存目录

    返回:
        tuple: (BM25索引, 页面文本列表)
    """
    import pickle

    os.makedirs(bm25_cache_dir, exist_ok=True)
    bm25_path = os.path.join(bm25_cache_dir, f"{dataset_name}_{doc_id}.bm25.pkl")
    texts_path = os.path.join(bm25_cache_dir, f"{dataset_name}_{doc_id}.texts.pkl")

    # 尝试加载已缓存的索引
    if os.path.exists(bm25_path) and os.path.exists(texts_path):
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)
        with open(texts_path, 'rb') as f:
            page_texts = pickle.load(f)
        return bm25_index, page_texts

    # 构建新的 BM25 索引
    pdf_path = f"/gz-data/dataset/{dataset_name}/{doc_id}.pdf"
    page_texts = extract_all_pages_text(pdf_path)

    # 过滤空页面
    valid_texts = [text for text in page_texts if text.strip()]
    valid_indices = [i for i, text in enumerate(page_texts) if text.strip()]

    if len(valid_texts) == 0:
        # 所有页面都为空，返回空索引
        with open(texts_path, 'wb') as f:
            pickle.dump(page_texts, f)
        return None, page_texts

    bm25_index = build_bm25_index(valid_texts)

    # 缓存索引
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25_index, f)
    with open(texts_path, 'wb') as f:
        pickle.dump(page_texts, f)

    return bm25_index, page_texts


def normalize_scores(scores, new_min=0.0, new_max=1.0):
    """
    将分数归一化到指定范围

    参数:
        scores: 原始分数数组
        new_min: 归一化后最小值
        new_max: 归一化后最大值

    返回:
        numpy.ndarray: 归一化后的分数
    """
    scores = np.array(scores)
    min_val = np.min(scores)
    max_val = np.max(scores)

    if max_val - min_val < 1e-9:
        return np.ones_like(scores) * new_min

    normalized = (scores - min_val) / (max_val - min_val)
    return normalized * (new_max - new_min) + new_min


def select_diverse_top_k(candidates, scores, k, similarity_threshold=0.5):
    """
    从候选中选择多样性 top-k 结果

    使用贪心策略：优先选择与已选结果相似度低的候选

    参数:
        candidates: 候选索引列表
        scores: 候选对应的分数 (越高越好)
        k: 需要选择的数量
        similarity_threshold: 相似度阈值，超过此阈值的候选不会被同时选择

    返回:
        list: 选中的 k 个候选索引
    """
    n_pages = max(candidates) + 1 if candidates else 0
    if n_pages == 0 or len(candidates) == 0:
        return []

    # 初始化相似度矩阵 (页面索引 -> 候选索引)
    page_to_idx = {c: i for i, c in enumerate(candidates)}

    selected = []
    remaining = candidates.copy()

    # 按分数排序，优先选择高分
    remaining.sort(key=lambda x: scores[page_to_idx[x]], reverse=True)

    while len(selected) < k and remaining:
        # 选择剩余中分数最高的
        best = remaining.pop(0)
        selected.append(best)

        # 移除与已选结果过于相似的候选 (简化版：只移除相同页面)
        # 完整实现需要计算页面间的语义相似度
        remaining = [c for c in remaining if c != best]

    return selected