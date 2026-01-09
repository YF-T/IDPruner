# -*- coding: utf-8 -*-
"""
================================================================================
|             DivPrune 剪枝策略 (`pruning/strategies/divprune.py`)             |
================================================================================
文件功能:
本文件实现了 DivPrune (Diversity-based Visual Token Pruning) 策略。

核心思想:
基于特征向量的多样性对视觉Token进行剪枝。
通过迭代选择与已选集合余弦距离最远（Minimum Distance Maximization）的Token，
来保证所选Token在特征空间中的覆盖率和多样性。

来源论文:
"DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models (CVPR 2025)"
"""
import torch
import torch.nn.functional as F
from typing import Dict, Any

# 从同级 utils 导入
from .utils import _extract_and_validate_vision_token_info


def divprune(context: Dict[str, Any], **kwargs) -> torch.Tensor:
    """
    DivPrune 策略函数。

    需要 `feature_map` 在 context 中。

    Args from kwargs:
        ratio (float): 要剪枝掉的视觉Token的比例，取值范围 [0.0, 1.0]。
                       (必须提供)
    """
    try:
        ratio = kwargs["ratio"]
    except KeyError:
        raise ValueError("`diversity_pruning` strategy requires 'ratio' in kwargs.")

    input_ids = context.get("input_ids")
    feature_map = context.get("feature_map")  # [B, Seq, Dim]

    if input_ids is None:
        raise ValueError("Context 中缺少 'input_ids'。")
    if feature_map is None:
        raise ValueError(
            "Diversity pruning requires 'feature_map' in the context, but it was not found."
        )

    bsz, seq_len, hidden_size = feature_map.shape
    device = input_ids.device

    assert bsz == 1, "Diversity pruning currently asserts batch_size=1"

    batch_keep_masks = []
    for i in range(bsz):
        input_ids_single = input_ids[i]
        # features_single shape: [Seq, Dim]
        features_single = feature_map[i]

        vision_indices, non_vision_indices, _, _ = (
            _extract_and_validate_vision_token_info(context)
        )

        num_vision_tokens = len(vision_indices)
        if num_vision_tokens == 0:
            batch_keep_masks.append(torch.ones_like(input_ids_single, dtype=torch.bool))
            continue

        num_to_keep = int(round(num_vision_tokens * (1 - ratio)))
        if ratio < 1.0 and num_to_keep == 0 and num_vision_tokens > 0:
            num_to_keep = 1
        if num_to_keep >= num_vision_tokens:
            batch_keep_masks.append(torch.ones_like(input_ids_single, dtype=torch.bool))
            continue

        # 提取视觉 token 特征 [N_vision, Dim]
        visual_feature_vectors = features_single[vision_indices]

        # 计算归一化特征和距离矩阵
        norm_matrix = F.normalize(visual_feature_vectors, p=2, dim=1)
        # print(f"norm_matrix shape: {norm_matrix.shape}")
        
        # 使用 PyTorch 计算距离矩阵更高效: 1 - Cosine Similarity
        # distance_matrix shape: [N_vision, N_vision]
        distance_matrix = 1.0 - torch.mm(norm_matrix, norm_matrix.t())
        
        # 将对角线设置为无穷大，避免选择自身
        distance_matrix.fill_diagonal_(float("inf"))

        kept_indices_local = torch.empty(num_to_keep, dtype=torch.long, device=device)

        # 初始分数：Maximin Initialization
        # 选择与所有其他点的最小距离最大的点作为起始点
        if num_vision_tokens > 1:
            min_dists_to_others, _ = torch.min(distance_matrix, dim=1)
            first_kept_idx = torch.argmax(min_dists_to_others)
        else:  # 只有一个视觉 token
            first_kept_idx = torch.tensor(0, device=device)

        kept_indices_local[0] = first_kept_idx
        num_kept = 1

        # 初始化当前已选点到所有点的最小距离向量 [N_vision]
        min_dists_to_kept = distance_matrix[:, first_kept_idx].clone()

        # 迭代选择 (最远点采样 FPS)
        while num_kept < num_to_keep:
            # 下一个要选的点是与当前已选集合最小距离最大的点
            scores = min_dists_to_kept.clone()
            
            # 排除已选点 (虽然 min_dists 逻辑上已选点的距离会变为0或inf，但为了保险)
            scores[kept_indices_local[:num_kept]] = -1.0 
            
            new_kept_idx = torch.argmax(scores)
            kept_indices_local[num_kept] = new_kept_idx
            num_kept += 1

            # 更新最小距离向量
            dist_to_new_kept = distance_matrix[:, new_kept_idx]
            min_dists_to_kept = torch.minimum(min_dists_to_kept, dist_to_new_kept)

        kept_vision_indices = vision_indices[kept_indices_local]

        final_kept_indices = torch.cat(
            [non_vision_indices.to(device), kept_vision_indices]
        )
        keep_mask = torch.zeros_like(input_ids_single, dtype=torch.bool)
        if final_kept_indices.numel() > 0:
            keep_mask[final_kept_indices] = True
        batch_keep_masks.append(keep_mask)

    return torch.stack(batch_keep_masks, dim=0)