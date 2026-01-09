# -*- coding: utf-8 -*-
"""
================================================================================
|       基于通用工具类的 Vision Selector 策略 (`selector_strategy.py`) v2.0     |
================================================================================
文件功能:
1. 实现基于 Vision Selector (TransformerScorer) 的全局剪枝策略。
2. 内部通过调用 `vision_selector_utils` 接口，自动兼容 V1 和 V2 版本的 Selector 模型。
3. 支持从特征空间提取权重，并执行确定性 Top-K 或 随机化采样 (Multinomial)。

核心改动:
- 彻底移除了手动加载和探测模型的逻辑，完全依赖 vision_selector_utils 提供的单例注册表缓存。
"""

import torch
from typing import Dict, Any, Optional
from loguru import logger as eval_logger

# 导入通用工具函数
from .utils import (
    _extract_and_validate_vision_token_info,
    get_valid_content_mask,
)
# [关键导入] 导入通用 Scorer 接口
from .vision_selector_utils import get_universal_selector_scores


def vision_selector_pruning(context: Dict[str, Any], **kwargs) -> torch.Tensor:
    """
    使用 Vision Selector 进行剪枝。

    Args (来自 kwargs):
        selector_path (str): 模型文件夹路径。
        ratio (float): 要剪掉的视觉 Token 比例 [0.0, 1.0]。
        randomize (bool): 是否对概率分布进行随机采样 (Multinomial)。
        text_selection_mode (str): V2 架构时文本提取模式 ('inverse' 或 'valid_content')。
        score_mode (str): 传递给 utils 的处理模式，默认 'soft_topk' (ratio=0.2)。

    Returns:
        torch.Tensor: [1, seq_len] 的布尔掩码。
    """
    # --- 1. 参数解析 ---
    try:
        selector_path = kwargs["selector_path"]
        pruning_ratio = kwargs["ratio"]
        randomize = kwargs.get("randomize", False)
        text_selection_mode = kwargs.get("text_selection_mode", "inverse")
        # 默认使用 soft_topk 模式获取权重分数，这与训练时的 0.2 比例约束一致
        score_mode = kwargs.get("score_mode", "soft_topk")
    except KeyError as e:
        raise ValueError(f"vision_selector_pruning 策略缺少必要参数: {e}")

    # --- 2. 获取上下文数据 ---
    input_ids = context.get("input_ids")
    inputs_embeds = context.get("inputs_embeds")
    if input_ids is None or inputs_embeds is None:
        raise ValueError("Context 中缺少 'input_ids' 或 'inputs_embeds'。")

    device = inputs_embeds.device
    bsz = inputs_embeds.shape[0]
    if bsz != 1:
        raise NotImplementedError("目前策略仅验证过 batch_size=1。")

    # --- 3. 定位模态 Token ---
    vision_indices, non_vision_indices, prunable_mask, _ = (
        _extract_and_validate_vision_token_info(context)
    )
    num_vision_tokens = len(vision_indices)

    if num_vision_tokens == 0:
        return torch.ones_like(input_ids, dtype=torch.bool)

    # 计算目标保留数量
    num_to_keep = int(round(num_vision_tokens * (1.0 - pruning_ratio)))
    if pruning_ratio < 1.0 and num_to_keep == 0:
        num_to_keep = 1
    if num_to_keep >= num_vision_tokens:
        return torch.ones_like(input_ids, dtype=torch.bool)

    # --- 4. 提取双模态特征 ---
    vision_hidden = inputs_embeds[:, vision_indices, :]
    
    text_hidden = None
    if text_selection_mode == "valid_content":
        valid_mask = get_valid_content_mask(input_ids[0])
        text_indices = torch.where(valid_mask)[0]
        if len(text_indices) > 0:
            text_hidden = inputs_embeds[:, text_indices, :]
    
    # 兜底：如果不需要 valid_content 或提取为空，使用所有非视觉 Token
    if text_hidden is None:
        text_indices = torch.where(~prunable_mask)[0]
        text_hidden = inputs_embeds[:, text_indices, :]

    # --- 5. 调用通用工具类获取分数 ---
    # 该函数会自动处理 V1/V2 架构探测，并返回 [B, N] 的重要性分数
    
    importance_weights = get_universal_selector_scores(
        selector_path=selector_path,
        vision_hidden=vision_hidden,
        text_hidden=text_hidden,
        mode=score_mode,
    )
    
    # 展平为 [N] 以便处理
    scores = importance_weights.squeeze(0).float()

    # --- 6. 执行离散化选择 (离散剪枝) ---
    k = min(num_to_keep, num_vision_tokens)
    
    if randomize:
        # 基于分数的概率采样
        probs = torch.softmax(scores, dim=-1)
        try:
            top_k_local_indices = torch.multinomial(probs, num_samples=k, replacement=False)
        except RuntimeError as e:
            # 严格原则：失败直接报错
            raise RuntimeError(f"Multinomial sampling failed with scores: {scores}. Error: {e}")
    else:
        # 确定性 Top-K
        _, top_k_local_indices = torch.topk(scores, k=k)

    # --- 7. 构建并返回全局 Mask ---
    kept_vision_global_indices = vision_indices[top_k_local_indices]
    final_kept_indices = torch.cat([non_vision_indices.to(device), kept_vision_global_indices])
    
    keep_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    keep_mask[0, final_kept_indices] = True

    return keep_mask