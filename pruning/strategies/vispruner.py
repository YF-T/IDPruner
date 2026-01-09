# -*- coding: utf-8 -*-
"""
================================================================================
|       VisPruner 剪枝策略实现 (`pruning/strategies/vispruner.py`) v2.2        |
================================================================================
文件功能:
本文件实现了 VisPruner 论文 (arXiv:2412.01818) 的核心两阶段剪枝算法。

算法流程:
1.  **阶段一 (重要性)**: 利用 `utils` 重计算的注意力图获取对齐后的重要性分数，选取 Top-K1 个 Token。
2.  **阶段二 (多样性)**: 针对剩余 Token，利用 LLM 空间的特征 (feature_map) 
    进行迭代匹配与去重 (Algorithm 1)，选取 Top-K2 个最具代表性的 Token。

核心特性:
- **特征统一性**: 无论模型架构（LLaVA/Qwen），统一在 LLM 特征空间执行多样性分析，
  确保特征顺序与 `utils` 返回的分数顺序严格对齐。
- **架构无关**: 依靠 `utils._recompute_attention_maps_for_all_images` 自动处理
  不同模型的空间聚合与顺序重排。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from loguru import logger as eval_logger

# 导入工具函数：依靠 utils 抹平模型差异
from .utils import (
    _extract_and_validate_vision_token_info,
    _recompute_attention_maps_for_all_images,
)


def vispruner_pruning(context: Dict[str, Any], **kwargs) -> torch.Tensor:
    """
    VisPruner 两阶段剪枝策略。
    [v2.3 修改] 引入 _regroup_tensors_by_count 以对齐物理分数段与逻辑特征段。
    """
    # --- 1. 参数解析 ---
    try:
        pruning_ratio = kwargs["ratio"]
        layer_idx = kwargs["layer_idx"]
        imp_ratio = kwargs.get("important_ratio_of_kept", 0.5)
        keep_ratio = 1.0 - pruning_ratio
    except KeyError as e:
        raise ValueError(f"vispruner_pruning 缺少必要参数: {e}")

    # --- 2. 基础数据获取与校验 ---
    input_ids = context.get("input_ids")
    feature_map = context.get("feature_map") 

    if input_ids is None or feature_map is None:
        raise ValueError("Context 缺少 'input_ids' 或 'feature_map'。请确保配置了 'need_feature_map': True")

    device = input_ids.device
    bsz = input_ids.shape[0]
    if bsz != 1:
        raise NotImplementedError("VisPruner 目前仅支持 batch_size=1。")

    # 获取视觉 Token 的全局分布及每张图的 Token 数量
    vision_indices_global, non_vision_indices_global, _, num_tokens_per_image = (
        _extract_and_validate_vision_token_info(context)
    )

    if len(vision_indices_global) == 0:
        return torch.ones_like(input_ids, dtype=torch.bool)

    # 获取 Q/K 用于分数重计算
    q_key = f"vit_post_rope_q_vit_{layer_idx}"
    k_key = f"vit_post_rope_k_vit_{layer_idx}"
    q_tensor = context.get(q_key)
    k_tensor = context.get(k_key)

    if q_tensor is None or k_tensor is None:
        raise ValueError(f"VisPruner 需要 ViT 第 {layer_idx} 层的 Q/K 用于重要性计算。")

    # 核心：重计算得到物理分段的分数列表
    final_scores_list, _ = _recompute_attention_maps_for_all_images(
        q_tensor, k_tensor, context
    )

    # --- [核心修改 1]: 导入并执行重组逻辑 ---
    from .utils import _regroup_tensors_by_count
    
    # 将物理分数的 [720, 720] 聚合为逻辑段的 [1440]
    final_scores_list, _ = _regroup_tensors_by_count(
        final_scores_list, num_tokens_per_image, None
    )
    # ----------------------------------------

    # --- 3. 准备特征与索引切分 ---
    all_kept_indices_global = []
    
    # 提取所有视觉 Token 在 LLM 空间的特征
    vision_features_total = feature_map[0, vision_indices_global, :]
    
    # 此时 final_scores_list 已与 num_tokens_per_image 对齐
    vision_features_list = torch.split(vision_features_total, num_tokens_per_image, dim=0)
    vision_indices_split = torch.split(vision_indices_global, num_tokens_per_image, dim=0)
    
    print(f"final_scores_list={[scores.shape for scores in final_scores_list]}")
    print(f"vision_features_list={[features.shape for features in vision_features_list]}")
    print(f"vision_indices_split={[indices.shape for indices in vision_indices_split]}")
    
    # --- 4. 按图片独立执行两阶段筛选 ---
    for img_idx, (scores, features, global_idx_map) in enumerate(
        zip(final_scores_list, vision_features_list, vision_indices_split)
    ):
        N = features.shape[0]
        if N == 0:
            all_kept_indices_global.append(global_idx_map)
            continue

        # scores 原本是 [1, N]，压缩到 [N]
        scores = scores.squeeze(0).float() 
        
        # 计算保留预算
        num_to_keep_total = int(round(N * keep_ratio))
        if keep_ratio > 0 and num_to_keep_total == 0:
            num_to_keep_total = 1
        
        if num_to_keep_total >= N:
            all_kept_indices_global.append(global_idx_map)
            continue
            
        num_imp = int(round(num_to_keep_total * imp_ratio))
        if num_to_keep_total > 1 and num_imp >= num_to_keep_total:
            num_imp = num_to_keep_total - 1
        num_div = num_to_keep_total - num_imp

        # --- 阶段一: 重要性选择 (Top-K) ---
        _, imp_indices_local = torch.topk(scores, k=num_imp)
        
        mask_residual = torch.ones(N, dtype=torch.bool, device=device)
        mask_residual[imp_indices_local] = False
        residual_indices_local = torch.where(mask_residual)[0]

        # --- 阶段二: 迭代匹配去重 (Algorithm 1) ---
        if num_div > 0 and len(residual_indices_local) > num_div:
            feat_norm = F.normalize(features.float(), p=2, dim=-1)
            
            while len(residual_indices_local) > num_div:
                R = len(residual_indices_local)
                r_batch = min(8, len(residual_indices_local) // 2, R - num_div)
                if r_batch <= 0:
                    break
                
                idx_a = residual_indices_local[::2]
                idx_b = residual_indices_local[1::2]
                
                if len(idx_a) == 0 or len(idx_b) == 0:
                    break
                    
                sim_matrix = torch.mm(feat_norm[idx_a], feat_norm[idx_b].t())
                max_sim_in_b, _ = sim_matrix.max(dim=-1) 
                
                sorted_a_rel_indices = max_sim_in_b.argsort(descending=True)
                keep_a_rel_indices = sorted_a_rel_indices[r_batch:]
                
                residual_indices_local = torch.cat([idx_a[keep_a_rel_indices], idx_b])
                residual_indices_local, _ = residual_indices_local.sort()

        # --- 合并结果 ---
        final_local_indices = torch.cat([imp_indices_local, residual_indices_local])
        final_local_indices = torch.unique(final_local_indices)
        
        if len(final_local_indices) > num_to_keep_total:
            final_local_indices = final_local_indices[:num_to_keep_total]
            
        all_kept_indices_global.append(global_idx_map[final_local_indices])

    # --- 5. 构建全序列布尔掩码 ---
    if not all_kept_indices_global:
        return torch.ones_like(input_ids, dtype=torch.bool)
        
    kept_indices_tensor = torch.cat(all_kept_indices_global)
    final_indices = torch.cat([non_vision_indices_global.to(device), kept_indices_tensor])
    
    keep_mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
    if final_indices.numel() > 0:
        keep_mask[final_indices] = True
    
    return keep_mask.unsqueeze(0)