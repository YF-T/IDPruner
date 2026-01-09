# -*- coding: utf-8 -*-
r"""
================================================================================
|    MMR + Vision Selector 策略 (`pruning/strategies/mmr_vision_selector.py`)  |
================================================================================
文件功能:
本文件实现了一种结合 MMR (Maximum Maximal Marginal Relevance) 和 Vision Selector (TransformerScorer)
的剪枝策略。

算法原理 (MMR):
MMR 是一种贪心算法，旨在同时最大化选定元素的相关性(Relevance)和多样性(Diversity)。
在每一轮迭代中，它选择满足以下公式的 Token $i$:
    $$ \text{argmax}_{i \in U} [ \lambda \cdot \text{Imp}(i) - (1 - \lambda) \cdot \max_{j \in S} \text{Sim}(i, j) ] $$

核心特性:
1.  **架构复用**: 调用 `vision_selector_utils` 接口，自动兼容 V1/V2 架构。
2.  **增量计算**: 使用向量化操作维护“当前候选点与已选集合的最大相似度”。
3.  **灵活性**: 支持自定义相似度和重要性计算函数。

使用方法:
    在配置中指定:
    "method": "mmr_pruning_vision_selector",
    "params": {
        "ratio": 0.5,
        "selector_path": "/path/to/your/selector/folder",
        "mmr_lambda": 0.5,
        ...
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import importlib.util
import time
import glob
import copy
from typing import Dict, Any, Tuple, Optional, Union, Callable
from loguru import logger as eval_logger

# 从同级 utils 导入
from .utils import _extract_and_validate_vision_token_info
# [关键导入] 导入通用 Scorer 接口
from .vision_selector_utils import get_universal_selector_scores

# ================================================================================
# |                      1. 默认计算逻辑与参数解析器                             |
# ================================================================================


def _default_similarity_compute(features: torch.Tensor) -> torch.Tensor:
    """
    默认相似度计算: 余弦相似度
    输入: [N, D] (特征矩阵)
    输出: [N, N] (相似度矩阵, 范围 [-1, 1])
    """
    # L2 Normalize
    features_norm = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
    # Matmul: X @ X.T
    return torch.matmul(features_norm, features_norm.t())


def _default_importance_compute(scores: torch.Tensor) -> torch.Tensor:
    """
    默认重要性计算: 归一化到 [0, 1] 区间
    MMR 对量级敏感，因此必须将重要性分数映射到与相似度（通常 [-1, 1] 或 [0, 1]）
    可比的范围内。
    """
    s_min = scores.min()
    s_max = scores.max()
    epsilon = 1e-6

    # 简单的线性映射到 [0, 1]
    norm_scores = (scores - s_min) / (s_max - s_min + epsilon)
    return norm_scores


def _resolve_callable(
    func_or_str: Union[Callable, str, None], default_func: Callable
) -> Callable:
    """
    解析函数参数。支持 Python 函数对象或代码字符串。
    """
    if func_or_str is None:
        return default_func

    if callable(func_or_str):
        return func_or_str

    if isinstance(func_or_str, str):
        local_scope = {}
        global_scope = {
            "torch": torch,
            "nn": torch.nn,
            "F": torch.nn.functional,
            "math": torch.math if hasattr(torch, "math") else None,
        }
        try:
            exec(func_or_str, global_scope, local_scope)
            if "compute" not in local_scope:
                raise ValueError(
                    "自定义代码字符串必须定义一个名为 'compute' 的函数作为入口。"
                )
            return local_scope["compute"]
        except Exception as e:
            raise ValueError(f"解析自定义代码字符串失败: {e}\n代码内容: {func_or_str}")

    raise ValueError(
        f"不支持的参数类型: {type(func_or_str)}。必须是 None, Callable 或 str。"
    )


# ================================================================================
# |                      2. MMR + Vision Selector 策略 (Batch Compatible)        |
# ================================================================================


def idpruner(context: Dict[str, Any], **kwargs) -> torch.Tensor:
    """
    结合 MMR 算法和 Vision Selector 的剪枝策略 (并行化改进版)。

    逻辑:
    1. 循环处理 Batch 中的每个样本。
    2. 对每个样本:
       a. 计算 Importance (Selector Output -> Normalized).
       b. 计算 Similarity Matrix (Cosine).
       c. 迭代选择: 每一轮并行选出当前 MMR 分数最高的 parallel_k 个点。
          Score = lambda * Imp - (1-lambda) * max_sim(candidate, selected).
    3. 堆叠结果并返回掩码。
    """
    # --- 1. 参数解析 ---
    try:
        ratio = kwargs["ratio"]
        selector_path = kwargs["selector_path"]
        mmr_lambda = kwargs.get("mmr_lambda", 0.5)  # 平衡系数
        parallel_k = kwargs.get("parallel_k", 1)    # [并行化修改] 每轮选择的数量

        sim_func_raw = kwargs.get("similarity_func")
        imp_func_raw = kwargs.get("importance_func")

        sim_func = _resolve_callable(sim_func_raw, _default_similarity_compute)
        imp_func = _resolve_callable(imp_func_raw, _default_importance_compute)

        if parallel_k < 1:
            raise ValueError(f"parallel_k 必须 >= 1，收到 {parallel_k}")

    except KeyError as e:
        raise ValueError(f"mmr_pruning_vision_selector 缺少必要参数: {e}")

    # --- 2. 获取上下文数据 ---
    input_ids = context.get("input_ids")
    inputs_embeds = context.get("inputs_embeds")

    if input_ids is None or inputs_embeds is None:
        raise ValueError("Context 缺少 'input_ids' 或 'inputs_embeds'。")

    bsz = inputs_embeds.shape[0]
    device = inputs_embeds.device

    batch_keep_masks = []

    # --- 3. 样本循环 ---
    for i in range(bsz):
        input_ids_single = input_ids[i]
        inputs_embeds_single = inputs_embeds[i]

        vision_indices, non_vision_indices, _, _ = (
            _extract_and_validate_vision_token_info(context)
        )
        N = len(vision_indices)

        # 默认掩码 (全保留)
        keep_mask_single = torch.ones_like(input_ids_single, dtype=torch.bool)

        if N > 0:
            num_to_keep = int(round(N * (1.0 - ratio)))
            if ratio < 1.0 and num_to_keep == 0:
                num_to_keep = 1

            if num_to_keep < N:
                keep_mask_single = torch.zeros_like(input_ids_single, dtype=torch.bool)
                keep_mask_single[non_vision_indices] = True

                # 4. 提取特征并计算基础分数
                visual_features = inputs_embeds_single[vision_indices, :]
                raw_scores = get_universal_selector_scores(
                    selector_path=selector_path,
                    vision_hidden=visual_features.unsqueeze(0),
                    text_hidden=None,
                    mode="raw"
                ).squeeze(0)

                importance = imp_func(raw_scores).float()
                similarity = sim_func(visual_features.float())

                # 5. [并行化核心] MMR 贪心选择循环
                selected_indices = []
                candidates_mask = torch.ones(N, dtype=torch.bool, device=device) # True 为待选
                max_sim_values = torch.full((N,), -2.0, device=device)

                while len(selected_indices) < num_to_keep:
                    # 计算当前步长 (最后一轮可能不足 parallel_k)
                    k_step = min(parallel_k, num_to_keep - len(selected_indices))
                    
                    if len(selected_indices) == 0:
                        # 第一轮：完全取决于重要性
                        mmr_score = importance.clone()
                    else:
                        # MMR 核心公式
                        mmr_score = (mmr_lambda * importance) - ((1 - mmr_lambda) * max_sim_values)

                    # 屏蔽已选点
                    mmr_score[~candidates_mask] = -float("inf")

                    # 并行选出当前分数最高的 k 个
                    _, best_batch_indices = torch.topk(mmr_score, k=k_step)

                    # 更新结果集
                    selected_indices.extend(best_batch_indices.tolist())
                    candidates_mask[best_batch_indices] = False

                    # 更新最大相似度维护向量
                    # 新的 max_sim = max(旧的 max_sim, 与本轮新选出的这批点的最大相似度)
                    # batch_sims shape: [N, k_step]
                    batch_sims = similarity[:, best_batch_indices]
                    batch_max_sim, _ = torch.max(batch_sims, dim=1)
                    max_sim_values = torch.maximum(max_sim_values, batch_max_sim)

                # 6. 构建最终掩码
                kept_local_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
                kept_global_indices = vision_indices[kept_local_indices]
                keep_mask_single[kept_global_indices] = True

        batch_keep_masks.append(keep_mask_single)

    return torch.stack(batch_keep_masks, dim=0)