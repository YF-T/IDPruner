# -*- coding: utf-8 -*-
"""
================================================================================
|             SCOPE 剪枝策略 (`pruning/strategies/scope.py`) v1.0            |
================================================================================
文件功能:
本文件实现了 SCOPE (Saliency-Coverage Oriented Token Pruning) 策略。
核心逻辑:
1. Saliency (显著性): 使用重计算的注意力分数作为先验重要性。
2. Coverage (覆盖率): 采用次模函数贪心优化，确保 Token 集合能最大化代表全图语义。

注意: 核心函数 SCOPE 严格保持官方提供的逻辑。
"""

import torch
import torch.nn.functional as F
import os
from typing import Dict, Any, Tuple, List

# 导入框架通用的工具函数
from .utils import (
    _extract_and_validate_vision_token_info,
    _recompute_attention_maps_for_all_images,
)

# ================================================================================
# |                   1. 官方 SCOPE 核心函数 (严格遵循原始逻辑)                    |
# ================================================================================

def SCOPE(visual_feature_vectors, num_selected_token, cls_attn=None, alpha=1.0):
    """
    Batched version of SCOPE that processes all batch elements simultaneously.
    Args:
        visual_feature_vectors: [B, N, D] batch of feature vectors
        num_selected_token: Number of tokens to select per batch
        cls_attn: [B, N] batch of attention weights
        alpha: Scaling factor for saliency scores (Passed as argument)
    Returns:
        selected_idx: [B, K] selected token indices for each batch
        cosine_simi: [B, N, N] batch of cosine similarity matrices
    """
    # Calculate cosine similarity for all batches at once
    norm_vectors = visual_feature_vectors / visual_feature_vectors.norm(dim=-1, keepdim=True)
    cosine_simi = torch.bmm(norm_vectors, norm_vectors.transpose(1, 2))
    
    B, N = visual_feature_vectors.shape[:2]
    device = visual_feature_vectors.device
    dtype = visual_feature_vectors.dtype
    
    # Pre-allocate tensors for all batches
    selected = torch.zeros(B, N, dtype=torch.bool, device=device)
    selected_idx = torch.empty(B, num_selected_token, dtype=torch.long, device=device)
    cur_max = torch.zeros(B, N, dtype=dtype, device=device)
    
    # Precompute cls_attn ** alpha for all batches
    # [修改]: alpha 现在作为函数参数传入
    if cls_attn is not None:
        cls_attn_powered = cls_attn ** alpha
    else:
        cls_attn_powered = torch.ones(B, N, dtype=dtype, device=device)
    
    for i in range(num_selected_token):
        # Calculate gains for all batches simultaneously
        unselected_mask = ~selected
        gains = torch.maximum(
            torch.zeros(1, dtype=dtype, device=device),
            cosine_simi.masked_fill(~unselected_mask.unsqueeze(1), 0) - 
            cur_max.unsqueeze(2)
        ).sum(dim=1)
        
        # Apply attention weights
        combined = os.environ.get('COMBINED', 'multi')
        if combined == 'multi':
            gains = gains * cls_attn_powered
        elif combined == 'add':
            gains = gains + cls_attn_powered
        else:
            raise NotImplementedError(f"Combined mode {combined} not supported")
            
        # Mask out already selected tokens
        gains = gains.masked_fill(~unselected_mask, float('-inf'))
        
        # Find best elements for all batches
        best_idx = gains.argmax(dim=1)
        
        # Update states for all batches
        selected[torch.arange(B, device=device), best_idx] = True
        selected_idx[:, i] = best_idx
        cur_max = torch.maximum(cur_max, cosine_simi[torch.arange(B, device=device), best_idx])
    
    return selected_idx, cosine_simi

# ================================================================================
# |                       2. prun_eval 框架包装器                                |
# ================================================================================

def scope_pruning(context: Dict[str, Any], **kwargs) -> torch.Tensor:
    """
    SCOPE 剪枝策略的框架入口函数。
    """
    # 1. 参数解析
    try:
        ratio = kwargs["ratio"]
        layer_idx = kwargs["layer_idx"]
        # [修改]: 显式接收 alpha 参数，默认 1.0
        alpha = kwargs.get("alpha", 1.0)
    except KeyError as e:
        raise ValueError(f"scope_pruning 缺少必要参数: {e}")

    # 2. 获取基础数据
    input_ids = context.get("input_ids")
    feature_map = context.get("feature_map")
    
    if input_ids is None or feature_map is None:
        raise ValueError("SCOPE 策略需要 'input_ids' 和 'feature_map'。")

    bsz = input_ids.shape[0]
    device = input_ids.device
    
    # 3. 提取视觉 Token 索引与信息
    vision_indices, non_vision_indices, _, _ = _extract_and_validate_vision_token_info(context)
    N_vision = len(vision_indices)
    
    if N_vision == 0:
        return torch.ones_like(input_ids, dtype=torch.bool)

    # 4. 重计算注意力得到显著性分数 (逻辑与 VisionZip 一致)
    q_key = f"vit_post_rope_q_vit_{layer_idx}"
    k_key = f"vit_post_rope_k_vit_{layer_idx}"
    q_tensor = context.get(q_key)
    k_tensor = context.get(k_key)
    
    if q_tensor is None or k_tensor is None:
        raise ValueError(f"SCOPE 需要 ViT 层 {layer_idx} 的 Q/K (Key: {q_key}, {k_key})。")

    # 使用框架重计算工具获取注意力图
    # 返回列表：[Image1_Score, Image2_Score, ...]
    final_scores_list, _ = _recompute_attention_maps_for_all_images(q_tensor, k_tensor, context)
    
    # 拼接显著性分数 [1, N_vision]
    cls_attn = torch.cat(final_scores_list, dim=1).to(device=device, dtype=feature_map.dtype)

    # 5. 准备算法输入并执行
    num_to_keep = int(round(N_vision * (1.0 - ratio)))
    if ratio < 1.0 and num_to_keep == 0 and N_vision > 0:
        num_to_keep = 1
        
    if num_to_keep >= N_vision:
        return torch.ones_like(input_ids, dtype=torch.bool)

    # 提取视觉特征 [B, N_vision, Dim]
    visual_features = feature_map[:, vision_indices, :]
    
    # 调用核心 SCOPE 函数
    selected_idx_local, _ = SCOPE(visual_features, num_to_keep, cls_attn=cls_attn, alpha=alpha)

    print(f"visual_features.shape: {visual_features.shape}, cls_attn.shape: {cls_attn.shape}")
    
    # 6. 映射索引并构建最终掩码
    # 针对 bsz=1 情况
    selected_idx_local = selected_idx_local.squeeze(0)
    kept_vision_global_indices = vision_indices[selected_idx_local]
    
    final_kept_indices = torch.cat([non_vision_indices.to(device), kept_vision_global_indices.to(device)])
    
    keep_mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
    if final_kept_indices.numel() > 0:
        keep_mask[final_kept_indices] = True
        
    return keep_mask.unsqueeze(0)