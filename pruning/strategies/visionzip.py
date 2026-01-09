# -*- coding: utf-8 -*-
"""
================================================================================
|       VisionZip 合并策略 (`pruning/strategies/visionzip.py`) v2.2            |
================================================================================
文件功能:
本文件实现了 VisionZip (Vision Zip Merging) 策略。
核心思想：
1.  **Dominant**: 基于重要性分数保留高分 Token。
2.  **Target**: 在剩余中均匀采样作为聚类中心 (根据 zip_ratio)。
3.  **Contextual**: 将其余 Token 根据特征相似度 (Key) 合并到最近的 Target Token 中。

v2.2 更新:
- [Bug修复] 修复了调用 `_recompute_attention_maps_for_all_images` 时参数过多的 TypeError。
- [逻辑优化] 废弃 `create_fake_mask_from_merging_results`，改为手动构建精确的 `fake_mask`。
  这确保了返回的 Mask 准确反映了保留节点（Dominant + Target）在原始序列中的位置，
  从而使 Adapter 在 `update_position_ids=False` 模式下能正确切片 Position IDs。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List
import math

# 导入工具函数
from .utils import (
    _recompute_attention_maps_for_all_images,
    _extract_and_validate_vision_token_info,
)
from .merging_utils import (
    get_dialogue_masks,
    # create_fake_mask_from_merging_results, # [v2.2] 已废弃，改为手动构建
)
from ..pruning_modules import get_context_key


def vision_zip_merging(
    context: Dict[str, Any], **kwargs
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VisionZip 合并策略函数。

    Args from kwargs:
        ratio (float): 总剪枝率 (1 - 保留率)。
        zip_ratio (float): 压缩目标率 (Target Token 占 **保留总数** 的比例)。
                           默认 10.0 / 64.0。
        layer_idx (int): ViT 层索引 (用于获取 Q/K)。
    """
    # --- 1. 参数解析 ---
    try:
        ratio = kwargs["ratio"]
        layer_idx = kwargs["layer_idx"]
        zip_ratio = kwargs.get("zip_ratio", 10.0 / 64.0)
    except KeyError as e:
        raise ValueError(f"vision_zip_merging 策略缺少必要参数: {e}")

    # 参数安全性检查
    keep_ratio = 1.0 - ratio

    # --- 2. 获取上下文数据 ---
    input_ids = context.get("input_ids")
    inputs_embeds = context.get("inputs_embeds")  # [1, Seq, Dim] (LLM 层级)

    if input_ids is None or inputs_embeds is None:
        raise ValueError("Context 缺少 'input_ids' 或 'inputs_embeds'。")

    # 获取 Q/K 用于重计算分数
    q_key = f"vit_post_rope_q_vit_{layer_idx}"
    k_key = f"vit_post_rope_k_vit_{layer_idx}"

    q_tensor_fine = context.get(q_key)
    k_tensor_fine = context.get(k_key)
    # cu_seqlens = context.get("cu_seqlens_full") # [v4.2] Removed

    if q_tensor_fine is None or k_tensor_fine is None:
        raise ValueError(
            f"VisionZip 需要 ViT 第 {layer_idx} 层的 Q/K 张量。\n"
            f"请检查配置中是否设置了 'needs': {{'need_vit_post_rope_qk': 'specific'}}\n"
            f"Expected keys: {q_key}, {k_key}"
        )

    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    # --- 3. 重计算并获取对齐后的 Scores 和 Keys ---
    # 返回列表：[Image1_Score, Image2_Score, ...]
    # [v2.2 Fix] 移除了多余的 cu_seqlens 参数
    final_scores_list, final_keys_list = _recompute_attention_maps_for_all_images(
        q_tensor_fine, k_tensor_fine, context
    )

    # --- 4. 准备分段信息 ---
    # _extract_and_validate_vision_token_info(context)

    (
        _,
        _,
        _,
        sizes_list_cpu,
        is_vision_list_cpu,
    ) = get_dialogue_masks(context)

    sizes_list_gpu = torch.tensor(sizes_list_cpu, device=device)
    is_vision_list_gpu = torch.tensor(is_vision_list_cpu, device=device)

    # 拆分 inputs_embeds
    hidden_split_list = torch.split(inputs_embeds, sizes_list_cpu, dim=1)

    # --- 5. 遍历处理每个片段 ---
    merge_weight_list = []
    fake_mask_segments = []  # [v2.2] 用来收集准确的 Mask

    # 迭代器
    scores_iter = iter(final_scores_list)
    keys_iter = iter(final_keys_list)

    for hidden_part, is_vision in zip(hidden_split_list, is_vision_list_cpu):
        # 5.1 文本片段 -> 无需处理
        if not is_vision:
            L_j = hidden_part.shape[1]
            fake_mask_segments.append(torch.ones(L_j, dtype=torch.bool, device=device))
            continue

        # 5.2 视觉片段 -> VisionZip 处理
        num_vision_tokens = hidden_part.shape[1]

        # 获取对应的 Score 和 Key
        try:
            # score: [1, N], key: [1, N, D]
            scores = next(scores_iter).squeeze(0)
            keys = next(keys_iter).squeeze(0)
        except StopIteration:
            raise RuntimeError("VisionZip: 图像片段数量不匹配 (Score/Key 耗尽)。")

        # 维度校验
        if scores.shape[0] != num_vision_tokens or keys.shape[0] != num_vision_tokens:
            raise ValueError(
                f"VisionZip 维度不匹配: hidden_part={num_vision_tokens}, "
                f"computed_score={scores.shape[0]}, computed_key={keys.shape[0]}"
            )

        if num_vision_tokens == 0:
            merge_weight_list.append(torch.empty(0, 0, dtype=dtype, device=device))
            fake_mask_segments.append(torch.tensor([], dtype=torch.bool, device=device))
            continue

        # --- VisionZip 核心逻辑 ---

        # A. 计算各部分数量
        num_to_keep_total = int(round(num_vision_tokens * keep_ratio))
        if keep_ratio > 0.0 and num_to_keep_total == 0 and num_vision_tokens > 0:
            num_to_keep_total = 1

        if num_to_keep_total >= num_vision_tokens:
            # 保留全部
            merge_weight_list.append(
                torch.eye(num_vision_tokens, dtype=dtype, device=device)
            )
            fake_mask_segments.append(
                torch.ones(num_vision_tokens, dtype=torch.bool, device=device)
            )
            continue

        # 计算 Target (背景聚类中心) 数量
        num_target = max(1, int(round(num_to_keep_total * zip_ratio)))

        # 计算 Dominant (重要前景) 数量
        num_dominant = num_to_keep_total - num_target

        # 再次检查 (理论上前面的 Assert 已经保证了，但防止取整误差)
        if num_dominant < 0:
            num_target = num_to_keep_total
            num_dominant = 0

        # B. 选择 Dominant Tokens (Top-K Importance)
        if num_dominant > 0:
            _, dominant_indices = torch.topk(scores, k=num_dominant)
        else:
            dominant_indices = torch.tensor([], dtype=torch.long, device=device)

        dominant_mask = torch.zeros(num_vision_tokens, dtype=torch.bool, device=device)
        if dominant_indices.numel() > 0:
            dominant_mask[dominant_indices] = True

        # C. 选择 Target Tokens (在非 Dominant 中均匀采样)
        candidate_indices = torch.where(~dominant_mask)[0]
        num_candidates = len(candidate_indices)

        if num_target > 0 and num_candidates > 0:
            # 均匀采样步长
            step = max(1, num_candidates // num_target)
            local_select = torch.arange(0, num_candidates, step, device=device)[
                :num_target
            ]
            target_indices = candidate_indices[local_select]
        else:
            target_indices = torch.tensor([], dtype=torch.long, device=device)

        target_mask = torch.zeros(num_vision_tokens, dtype=torch.bool, device=device)
        if target_indices.numel() > 0:
            target_mask[target_indices] = True

        # D. 确定 Contextual Tokens (被合并的)
        contextual_mask = ~(dominant_mask | target_mask)
        contextual_indices = torch.where(contextual_mask)[0]

        # E. 构建合并矩阵 T [M, N]
        kept_indices = torch.cat([dominant_indices, target_indices]).sort().values

        # [v2.2 Change] 生成精确的 Segment Mask (对应于实际保留的节点位置)
        # 这确保了后续 Position ID 切片时保留的是 Dominant 和 Target 的位置信息
        segment_mask = torch.zeros(num_vision_tokens, dtype=torch.bool, device=device)
        if kept_indices.numel() > 0:
            segment_mask[kept_indices] = True
        fake_mask_segments.append(segment_mask)

        # 映射: Old Index -> New Matrix Row Index
        index_map = torch.full(
            (num_vision_tokens,), -1, dtype=torch.long, device=device
        )
        index_map[kept_indices] = torch.arange(len(kept_indices), device=device)

        merge_mat = torch.zeros(
            len(kept_indices), num_vision_tokens, dtype=dtype, device=device
        )

        # 1. 填充保留 Token (Identity)
        merge_mat.scatter_(1, kept_indices.unsqueeze(1), 1.0)

        # 2. 填充 Contextual Token (Merge to Nearest Target)
        if len(contextual_indices) > 0 and len(target_indices) > 0:
            ctx_feats = F.normalize(keys[contextual_indices].float(), p=2, dim=1)
            tgt_feats = F.normalize(keys[target_indices].float(), p=2, dim=1)

            # Similarity
            sim = torch.mm(ctx_feats, tgt_feats.t())

            # 找最近 Target
            _, best_target_local_idx = torch.max(sim, dim=1)
            best_target_global_idx = target_indices[best_target_local_idx]

            # 找矩阵行号
            target_row_indices = index_map[best_target_global_idx]

            # 填入矩阵 (Accumulate)
            indices_to_add = torch.stack([target_row_indices, contextual_indices])
            values_to_add = torch.ones(
                len(contextual_indices), dtype=dtype, device=device
            )
            merge_mat.index_put_(tuple(indices_to_add), values_to_add, accumulate=True)

        # 3. 归一化 (Average Pooling)
        row_sums = merge_mat.sum(dim=1, keepdim=True)
        merge_mat = merge_mat / (row_sums + 1e-8)

        merge_weight_list.append(merge_mat.to(dtype))

    # --- 6. 生成 Fake Mask (直接拼接) ---
    fake_mask = torch.cat(fake_mask_segments, dim=0)

    # print(f"VisionZip: merge_weight_list={merge_weight_list}")
    # print(f"VisionZip: fake_mask={fake_mask}")
    # print(f"VisionZip: sizes_list_gpu={sizes_list_gpu}")
    # print(f"VisionZip: is_vision_list_gpu={is_vision_list_gpu}")

    return (merge_weight_list, sizes_list_gpu, is_vision_list_gpu, fake_mask)
