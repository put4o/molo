📋 文档 RAG 任务完整流程伪代码
概述
这是一个基于视觉语言模型（VLM）的文档检索系统，使用 ColPali 进行向量索引，结合 Qwen-3B 进行逻辑感知的页面检索。

第一阶段：索引 (Indexing)
┌─────────────────────────────────────────────────────────────┐
│                    索引阶段流程                              │
├─────────────────────────────────────────────────────────────┤
│  加载配置参数 → 初始化ColPali模型 → 扫描PDF文件列表          │
│        ↓                                                    │
│  遍历每个PDF文档 ──→ [已存在嵌入?] ─是→ 跳过                 │
│        ↓ 否                                                 │
│  PDF转图片(144 DPI) → 分批编码 → 存储嵌入向量                │
└─────────────────────────────────────────────────────────────┘
# ============================================================
# 阶段一：索引构建 - PDF文档向量化
# 入口命令: python3 index.py --dataset MMLong --save_img
# ============================================================

PROCEDURE index_documents(dataset_name="MMLong"):
    
    # ─────────────────────────────────────────────────────────
    # 步骤 1: 解析命令行参数并初始化环境
    # ─────────────────────────────────────────────────────────
    
    SET save_dir       = "/gz-data/tmp/tmp_embs/{dataset_name}"      # 嵌入保存路径
    SET img_save_dir   = "/gz-data/tmp/tmp_imgs/{dataset_name}"      # 图片保存路径
    SET batch_size     = 32                                          # 编码批大小
    SET model_name     = "vidore/colpali"                            # 预训练模型名称
    SET device         = "cuda:0"                                    # 计算设备
    SET resolution     = 144                                         # PDF转图片分辨率(DPI)
    
    # ─────────────────────────────────────────────────────────
    # 步骤 2: 初始化 ColPali 模型和处理器
    # ─────────────────────────────────────────────────────────
    
    # ColPali = ColQwen2 + PaliGemma 的视觉语言模型
    # 用于将图像编码为高维向量表示
    
    model = ColPali.from_pretrained(
        model_name,                           # 加载预训练权重
        torch_dtype=torch.bfloat16,           # 使用bfloat16节省显存
        device_map=device                     # 分配到GPU设备
    )
    model.eval()                              # 设置为评估模式(不更新梯度)
    
    processor = ColPaliProcessor.from_pretrained(model_name)
    # 处理器负责:
    #   - 将PIL图像预处理为模型输入格式
    #   - 将文本查询编码为向量
    #   - 计算多向量相似度分数
    
    # ─────────────────────────────────────────────────────────
    # 步骤 3: 扫描数据集目录，获取所有PDF文件列表
    # ─────────────────────────────────────────────────────────
    
    pdf_dir = "/gz-data/dataset/{dataset_name}"
    pdf_files = prepare_files(pdf_dir, suffix=".pdf")
    # prepare_files() 实现:
    #   RETURN [file for file in os.listdir(pdf_dir) if file.endswith(".pdf")]
    
    CREATE_DIR_IF_NOT_EXISTS(save_dir)
    CREATE_DIR_IF_NOT_EXISTS(img_save_dir)
    
    # ─────────────────────────────────────────────────────────
    # 步骤 4: 遍历每个PDF文档，生成向量化嵌入
    # ─────────────────────────────────────────────────────────
    
    FOR each pdf_file IN tqdm(pdf_files, desc="Encoding PDFs..."):
        
        doc_id = pdf_file.replace(".pdf", "")           # 提取文档ID (不含扩展名)
        doc_path = "{pdf_dir}/{pdf_file}"               # 完整PDF路径
        
        # 检查是否已存在嵌入，避免重复计算
        IF os.path.exists("{save_dir}/{doc_id}.pt"):
            PRINT "Embeddings for {pdf_file} already exists. Skipping..."
            CONTINUE
        
        # ─────────────────────────────────────────────────────────
        # 子步骤 4.1: 将PDF转换为页面快照图像
        # ─────────────────────────────────────────────────────────
        
        # 使用 pdf2image 库将PDF的每一页转换为PNG图像
        page_images = convert_from_path(
            doc_path, 
            dpi=resolution                              # 设置分辨率144 DPI
        )
        # convert_from_path() 实现:
        #   RETURN [PIL.Image.Image, ...]  # 每一页的图像列表
        
        # 如果指定了 --save_img 参数，保存图像用于后续VLM分析
        IF save_img_flag:
            FOR page_num, page_snapshot IN ENUMERATE(page_images):
                img_filename = "{doc_id}-{page_num+1}.png"   # 页码从1开始
                IF NOT os.path.exists("{img_save_dir}/{img_filename}"):
                    page_snapshot.save("{img_save_dir}/{img_filename}")
                    # 保存为PNG格式，保留高质量图像用于VLM处理
        
        # ─────────────────────────────────────────────────────────
        # 子步骤 4.2: 分批将图像编码为向量嵌入
        # ─────────────────────────────────────────────────────────
        
        total_image_embeds = torch.Tensor().to(device)   # 初始化空张量存储嵌入
        
        FOR idx IN RANGE(0, len(page_images), batch_size):
            
            # 4.2.1: 获取当前批次的图像
            batch_images = page_images[idx : idx + batch_size]
            
            # 4.2.2: 使用处理器预处理图像
            # - 调整图像大小至模型要求的分辨率
            # - 归一化像素值
            # - 转换为PyTorch张量
            batch_input = processor.process_images(batch_images)
            batch_input = batch_input.to(device)
            
            # 4.2.3: 清理GPU显存缓存
            WITH torch.cuda.device(device):
                torch.cuda.empty_cache()
            
            # 4.2.4: 前向传播生成图像嵌入
            WITH torch.no_grad():                           # 关闭梯度计算
                image_embeds = model(**batch_input)
                # 输出形状: [batch_size, hidden_dim, seq_len]
                # ColPali使用多向量输出表示图像的不同区域
            
            # 4.2.5: 拼接当前批次的嵌入到总嵌入
            total_image_embeds = torch.cat(
                (total_image_embeds, image_embeds), 
                dim=0                                        # 按批次维度拼接
            )
        
        # ─────────────────────────────────────────────────────────
        # 子步骤 4.3: 保存文档嵌入到磁盘
        # ─────────────────────────────────────────────────────────
        
        embed_path = "{save_dir}/{doc_id}.pt"
        torch.save(total_image_embeds, embed_path)
        PRINT "Save Embeddings {total_image_embeds.shape} to {embed_path}"
        
        # 嵌入张量形状说明:
        # [num_pages, hidden_dim, patch_seq_len]
        # 例如: [50, 128, 64] 表示50页文档, 128维特征, 64个图像块
    
    PRINT "Indexing Complete!"
    
END PROCEDURE


第二阶段：检索 (Retrieving)
流程图
┌──────────────────────────────────────────────────────────────────────┐
│                        检索阶段流程                                   │
├──────────────────────────────────────────────────────────────────────┤
│  加载文档嵌入 → 构建页面图(beamsearch) → 加载Qwen-3B模型                │
│        ↓                                                             │
│  遍历每个查询样本                                                      │
│        ↓                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Base方法:                                                        │ │
│  │   计算查询向量 → 与所有页面计算相似度 → 返回Top-K页面               │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Beamsearch方法 (VLM增强):                                        │ │
│  │   初始Beam选择 → VLM评估相关性 → 页面图扩展 → 迭代搜索              │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│        ↓                                                             │
│  保存检索结果到JSON文件                                               │
└──────────────────────────────────────────────────────────────────────┘
# ============================================================
# 阶段二：检索 - 逻辑感知的页面检索
# 入口命令: python3 retrieve.py --dataset MMLong --method beamsearch
# ============================================================

# ─────────────────────────────────────────────────────────
# 全局配置参数
# ─────────────────────────────────────────────────────────

ARGUMENTS:
    dataset     = "MMLong"                    # 数据集名称
    method      = "beamsearch"                # 检索方法: "base" 或 "beamsearch"
    encoder     = "vidore/colpali"            # 编码器模型
    emb_root    = "/gz-data/tmp/tmp_embs"     # 嵌入文件根目录
    top_k       = 20                          # Base方法返回的Top-K结果
    threshold   = 0.3                         # 相似度阈值(用于beamsearch)
    alpha       = 0.4                         # 相似度权重: alpha*sim + (1-alpha)*vlm
    beam_width  = 3                           # Beamsearch宽度
    max_hop     = 4                           # 最大搜索跳数
    model_name  = "QwenVL-3B-lora"            # VLM模型名称

# ─────────────────────────────────────────────────────────
# 辅助函数定义
# ─────────────────────────────────────────────────────────

FUNCTION query_vlm_relevance(query, doc_info, vlm_model):
    """
    使用Qwen-3B VLM评估单个页面与查询的相关性
    
    参数:
        query: 用户查询文本
        doc_info: 元组 (doc_id, page_num)
        vlm_model: 已加载的Qwen-VL模型
    
    返回:
        relevance_score: 1-5的整数，表示页面相关性
    """
    
    doc_id, page_num = doc_info
    
    # 步骤1: 确保页面图像存在
    img_path = "/gz-data/tmp/tmp_imgs/{args.dataset}/{doc_id}-{page_num}.png"
    IF NOT os.path.exists(img_path):
        # 如果图像不存在，从PDF重新提取该页
        page_image = convert_from_path(
            pdf_path="/gz-data/dataset/{args.dataset}/{doc_id}.pdf",
            first_page=page_num,
            last_page=page_num,
            dpi=144
        )[0]
        page_image.save(img_path, "PNG")
    
    # 步骤2: 生成相关性评估提示词
    IF args.dataset == "MMLong":
        prompt = generate_relevance_prompt(query)
    ELSE:
        prompt = generate_relevance_prompt_detailed(query)
    
    # prompt内容示例:
    # """
    # # GOAL #
    # You are a Retrieval Expert, evaluate page relevance to query.
    # Rate 1-5:
    # - 5: Highly relevant - contains complete information
    # - 4: Very relevant - contains most information  
    # - 3: Moderately relevant - contains some useful information
    # - 2: Slightly relevant - minor connection
    # - 1: Irrelevant - no related information
    # # QUERY #
    # {query}
    # Provide just a single number (1-5).
    # """
    
    # 步骤3: 调用Qwen-VL模型进行推理
    response = get_response_concat(
        vlm_model,
        prompt,
        img_path,
        max_new_tokens=16,          # 只需要返回单个数字
        temperature=1.0
    )
    # get_response_concat() 实现:
    #   1. 构建消息: [用户角色, 包含图像URL和文本提示]
    #   2. 应用聊天模板
    #   3. 处理视觉信息
    #   4. 模型生成响应
    #   5. 解码并清理特殊token
    
    # 步骤4: 解析响应提取分数
    score_match = REGEX_SEARCH(r'[1-5]', response)   # 提取第一个1-5的数字
    IF score_match:
        relevance_score = INT(score_match.group(0))
    ELSE:
        relevance_score = 3                         # 默认中等相关
    
    RETURN relevance_score


# ─────────────────────────────────────────────────────────
# 文档检索器类定义
# ─────────────────────────────────────────────────────────

CLASS DocumentRetriever:
    """
    统一文档检索类，支持多种检索策略
    """
    
    CONSTRUCTOR(encoder, processor, device, batch_size=512):
        self.encoder   = encoder          # ColPali编码器
        self.processor = processor        # 处理器
        self.device    = device           # 计算设备
        self.batch_size = batch_size      # 批处理大小
    
    FUNCTION compute_scores(query, all_embeds):
        """
        计算查询与所有页面的相似度分数
        """
        
        # 步骤1: 将查询文本编码为向量
        queries = processor.process_queries(queries=[query])
        queries = queries.to(device)
        query_embeds = encoder(**queries)
        # query_embeds形状: [1, hidden_dim, query_seq_len]
        
        # 步骤2: 分批计算与所有页面的多向量相似度
        all_scores = []
        
        FOR idx IN RANGE(0, all_embeds.shape[0], self.batch_size):
            
            batch_embeds = all_embeds[idx : idx + self.batch_size]
            batch_embeds = FloatTensor(batch_embeds).to(
                device=device,
                dtype=query_embeds.dtype
            )
            
            WITH torch.no_grad():
                # 多向量相似度计算 (ColPali特有)
                # 对query的每个向量与page的每个向量计算点积
                tmp_scores = processor.score_multi_vector(
                    query_embeds,      # [1, hidden_dim, q_seq]
                    batch_embeds       # [batch, hidden_dim, p_seq]
                )
                # 输出形状: [batch_size, query_seq, page_seq]
                # 需要根据形状进行处理
                IF len(tmp_scores.shape) > 1:
                    tmp_scores = tmp_scores[0]   # 取第一个维度
            
            all_scores.append(tmp_scores)
        
        # 步骤3: 合并所有分数
        scores = torch.cat(all_scores, dim=0).cpu()
        
        # 清理内存
        DEL all_scores, queries, query_embeds
        
        RETURN scores
    
    FUNCTION base_retrieve(query, all_embeds, top_k=10):
        """
        基础检索方法：直接基于向量相似度
        """
        
        # 计算所有页面的相似度分数
        scores = compute_scores(query, all_embeds)
        
        # 排序并返回Top-K结果
        top_indices = scores.argsort(dim=-1, descending=True)[:top_k]
        top_scores = scores[top_indices].tolist()
        
        # 页码从1开始计数
        RETURN [idx + 1 FOR idx IN top_indices], top_scores
    
    FUNCTION vlm_retrieve(query, all_embeds, graph, doc_id, 
                         beam_width=3, max_hop=5, verbose=True):
        """
        VLM增强的Beamsearch检索方法
        
        参数:
            query: 用户查询
            all_embeds: 文档所有页面的嵌入向量
            graph: 页面邻接图 {page_idx: [neighbor_idx, ...]}
            doc_id: 文档ID
            beam_width: 每轮保留的候选数量
            max_hop: 最大搜索跳数
            verbose: 是否打印详细信息
        """
        
        # 步骤1: 计算初始相似度分数
        scores = compute_scores(query, all_embeds)
        
        # 步骤2: 归一化分数到[0, 1]范围
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()
        score_range = max_score - min_score IF max_score > min_score ELSE 1.0
        
        score_dict = {
            i: (scores[i].item() - min_score) / score_range 
            FOR i IN RANGE(scores.shape[0])
        }
        
        # 步骤3: 初始化Beam
        # 选择相似度最高的beam_width个页面作为初始候选
        initial_beam = scores.argsort(dim=-1, descending=True)[:beam_width]
        initial_beam = initial_beam.tolist()
        
        visited = SET(initial_beam)              # 记录已访问的页面
        vlm_score_cache = {}                     # 缓存VLM评估结果
        vlm_query_times = 0                      # VLM调用次数计数
        
        # 步骤4: 评估初始Beam中每个页面的VLM相关性
        FOR node IN initial_beam:
            
            # 调用Qwen-VL评估页面相关性
            vlm_score = query_vlm_relevance(
                query, 
                (doc_id, node + 1),     # 页码从1开始
                vlm_model
            )
            
            vlm_query_times += 1
            vlm_score_cache[node] = vlm_score
            
            # 归一化VLM分数到[0, 1]
            norm_vlm_score = (vlm_score - 1.0) / 4.0
            
            # 综合分数 = alpha * 相似度 + (1-alpha) * VLM分数
            combined_score = args.alpha * score_dict[node] + \
                            (1.0 - args.alpha) * norm_vlm_score
            score_dict[node] = combined_score
        
        IF verbose:
            PRINT f"Initial Beam: {[n+1 FOR n IN initial_beam]}"
            PRINT f"Initial Scores: {[round(score_dict[n], 3) FOR n IN initial_beam]}"
        
        # 记录当前最优结果
        result_dict = {node: score_dict[node] FOR node IN initial_beam}
        
        # 步骤5: 开始Beamsearch迭代
        FOR hop IN RANGE(max_hop):
            
            candidates = []        # 候选邻居页面
            
            FOR node IN current_beam:
                # 获取当前节点的邻居页面
                neighbor_pages = graph.get(node, [])
                
                FOR neighbor IN neighbor_pages:
                    IF neighbor NOT IN visited:
                        
                        # 标记为已访问
                        visited.add(neighbor)
                        
                        # VLM评估邻居页面
                        vlm_score = query_vlm_relevance(
                            query,
                            (doc_id, neighbor + 1),
                            vlm_model
                        )
                        vlm_query_times += 1
                        vlm_score_cache[neighbor] = vlm_score
                        
                        norm_vlm_score = (vlm_score - 1.0) / 4.0
                        
                        # 计算综合分数
                        combined_score = args.alpha * score_dict[neighbor] + \
                                        (1.0 - args.alpha) * norm_vlm_score
                        score_dict[neighbor] = combined_score
                        
                        candidates.append((neighbor, combined_score))
                        result_dict[neighbor] = combined_score
            
            # 如果没有新候选，退出搜索
            IF NOT candidates:
                BREAK
            
            # 步骤6: 选择Top-K候选作为下一轮Beam
            candidates = SORTED(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            current_beam = [node FOR node, _ IN candidates]
            
            IF verbose:
                PRINT f"Hop {hop+1}: Beam = {[n+1 FOR n IN current_beam]}"
                PRINT f"Hop {hop+1}: Scores = {[round(score_dict[n], 3) FOR n IN current_beam]}"
        
        # 步骤7: 过滤和排序最终结果
        final_results = [
            (node, score) FOR node, score IN result_dict.items()
            IF score >= threshold
        ]
        final_results = SORTED(final_results, key=lambda x: x[1], reverse=True)
        
        # 提取页面编号和分数
        evidence_pages = [node + 1 FOR node, _ IN final_results]
        page_scores = [score FOR _, score IN final_results]
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        PRINT f"Total Pages: {all_embeds.shape[0]}"
        PRINT f"VLM Query Times: {vlm_query_times}"
        
        RETURN evidence_pages, page_scores


# ─────────────────────────────────────────────────────────
# 主程序入口
# ─────────────────────────────────────────────────────────

PROCEDURE main_retrieval():
    
    # 步骤1: 初始化设备
    device = torch.device("cuda:0")
    
    # 步骤2: 加载所有文档的嵌入向量
    emb_root = "/gz-data/tmp/tmp_embs/{dataset}"
    doc2emb = load_all_doc_embeddings(emb_root)
    # load_all_doc_embeddings() 实现:
    #   FOR each .pt file IN emb_root:
    #       embeds = torch.load(file, map_location="cpu")
    #       doc_id = filename.replace(".pt", "")
    #       doc2emb[doc_id] = embeds.detach().numpy()
    #   RETURN doc2emb  # {doc_id: numpy_array}
    
    # 步骤3: 初始化ColPali编码器
    encoder = ColPali.from_pretrained(
        encoder_model,
        torch_dtype=torch.bfloat16,
        device_map=device
    ).eval()
    
    processor = ColPaliProcessor.from_pretrained(encoder_model)
    retriever = DocumentRetriever(
        encoder=encoder,
        processor=processor,
        device=device
    )
    
    # 步骤4: 加载查询样本
    samples = json.load(open("/gz-data/dataset/samples_{dataset}.json", "r"))
    # samples结构:
    # [
    #   {
    #     "question": "...",
    #     "doc_id": "xxx.pdf",
    #     "evidence_pages": [5, 12, 23],  # 可选的ground truth
    #     ...
    #   },
    #   ...
    # ]
    
    # 步骤5: 如果使用beamsearch方法，额外初始化
    IF method == "beamsearch":
        
        # 5.1: 加载Qwen-3B VLM模型
        vlm_model = init_model(model_name, device)
        # init_model() 实现:
        #   model_path = "/gz-data/models"
        #   model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #       model_path,
        #       torch_dtype=torch.bfloat16,
        #       device_map=device
        #   ).eval()
        #   processor = AutoProcessor.from_pretrained(model_path)
        #   RETURN model
        
        # 5.2: 构建所有文档的页面邻接图
        doc2graph = {}
        
        FOR doc_id, doc_emb IN tqdm(doc2emb.items(), desc="Constructing Page Graph"):
            
            graph = construct_page_graph(doc_emb, threshold=0.8, sim_measure="cosine")
            # construct_page_graph() 实现:
            #   n_pages = doc_emb.shape[0]
            #   IF n_pages <= 3: RETURN None
            #   
            #   # 计算所有页面两两之间的相似度
            #   sim_matrix = np.zeros((n_pages, n_pages))
            #   FOR i IN range(n_pages):
            #       FOR j IN range(i+1, n_pages):
            #           vec_i = doc_emb[i]   # [hidden_dim, seq_len]
            #           vec_j = doc_emb[j]
            #           sim = compute_embed_similarity(vec_i, vec_j, "cosine")
            #           sim_matrix[i][j] = sim
            #           sim_matrix[j][i] = sim
            #   
            #   # 构建KNN图 (每个节点连接top-k相似邻居)
            #   page_graph = defaultdict(list)
            #   k_value = 5
            #   threshold = 0.8
            #   FOR i IN range(n_pages):
            #       top_k_idx = np.argsort(sim_matrix[i])[::-1][:k_value]
            #       FOR j IN top_k_idx:
            #           IF sim_matrix[i][j] >= threshold:
            #               page_graph[i].append(j)
            #               page_graph[j].append(i)
            #   
            #   RETURN page_graph  # {page_idx: [neighbor_idx, ...]}
            
            doc2graph[doc_id] = deepcopy(graph)
    
    # 步骤6: 遍历每个查询样本进行检索
    FOR sample IN tqdm(samples, desc="Retrieving"):
        
        query = sample["question"]
        target_doc = sample["doc_id"].replace(".pdf", "")
        target_doc_embedding = doc2emb[target_doc]
        
        IF method == "base":
            # 基础检索：纯向量相似度
            ranked_pages, page_scores = retriever.base_retrieve(
                query,
                target_doc_embedding,
                top_k=top_k
            )
            
        ELIF method == "beamsearch":
            # VLM增强检索：结合向量相似度和语义理解
            target_graph = doc2graph.get(target_doc, defaultdict(list))
            
            ranked_pages, page_scores = retriever.vlm_retrieve(
                query,
                target_doc_embedding,
                target_graph,
                target_doc,
                beam_width=beam_width,
                max_hop=max_hop,
                verbose=beam_verbose
            )
        
        # 步骤7: 保存检索结果
        sample["pages_ranking"] = str(ranked_pages)
        sample["pages_scores"] = str(page_scores)
        
        IF "evidence_pages" IN sample:
            PRINT f"Ground-truth: {sample['evidence_pages']}")
        
        PRINT f"Prediction: {ranked_pages[:5]}")
        
        # 实时保存到JSON文件
        output_file = "/gz-data/dataset/retrieved/samples_{dataset}_{method}{vlm_suffix}.json"
        CREATE_DIR_IF_NOT_EXISTS(os.path.dirname(output_file))
        json.dump(samples, open(output_file, "w"), indent=4)
    
    PRINT "Retrieval Complete!"

END PROCEDURE

流程总结图
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MoLoRAG 完整流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────┐         ┌────────────────────┐                      │
│  │  Step 1: Indexing  │         │ Step 2: Retrieving │                      │
│  └─────────┬──────────┘         └─────────┬──────────┘                      │
│            │                              │                                  │
│            ▼                              ▼                                  │
│  ┌────────────────────┐         ┌────────────────────┐                      │
│  │ PDF Documents      │         │ User Query         │                      │
│  └─────────┬──────────┘         └─────────┬──────────┘                      │
│            │                              │                                  │
│            ▼                              ▼                                  │
│  ┌────────────────────┐         ┌────────────────────┐                      │
│  │ pdf2image          │         │ ColPali Encoder    │                      │
│  │ (144 DPI)          │         │ Query → Vector     │                      │
│  └─────────┬──────────┘         └─────────┬──────────┘                      │
│            │                              │                                  │
│            ▼                              ▼                                  │
│  ┌────────────────────┐         ┌────────────────────┐                      │
│  │ ColPali Model      │         │ Similarity Search  │                      │
│  │ Image → Embeddings │         │ Base Method        │                      │
│  └─────────┬──────────┘         └────────────────────┘                      │
│            │                                                               │
│            ▼                                                               │
│  ┌────────────────────┐                                                    │
│  │ Save to .pt Files  │                                                    │
│  │ [pages, dim, seq]  │                                                    │
│  └────────────────────┘                                                    │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│                              Beamsearch 流程                                 │
│                                                                             │
│  ┌────────────────────┐    ┌────────────────────┐                          │
│  │ Page Graph         │    │ Qwen-3B VLM        │                          │
│  │ (KNN, cos>0.8)     │    │ Relevance Scoring  │                          │
│  └─────────┬──────────┘    └────────────────────┘                          │
│            │                              │                                  │
│            └──────────┬───────────────────┘                                  │
│                       ▼                                                     │
│            ┌────────────────────┐                                            │
│            │ Beam Search Loop   │                                            │
│            │ 1. Select Top-K    │◄─────────────────┐                         │
│            │ 2. VLM Evaluation  │                  │                         │
│            │ 3. Graph Expansion │                  │                         │
│            │ 4. Re-rank         │                  │                         │
│            └────────────────────┘                  │                         │
│                       │                             │                         │
│                       ▼                             │                         │
│            ┌────────────────────┐                   │                         │
│            │ Evidence Pages     │<──────────────────┘                         │
│            │ + Scores           │        (until max_hop)                      │
│            └────────────────────┘                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘



🔑 关键组件说明
组件	作用	关键技术
ColPali	多模态文档嵌入模型	基于PaliGemma，将图像转换为高维向量
ColPaliProcessor	预处理/后处理	图像处理、查询编码、多向量相似度计算
Qwen-3B	视觉语言模型	理解文档页面内容，评估相关性(1-5分)
Page Graph	页面邻接图	基于相似度的KNN图，支持Beamsearch扩展
Multi-vector Similarity	多向量匹配	处理文档中的多区域、多块内容匹配