# -*- coding: utf-8 -*-
"""
================================================================================
|                DART 剪枝策略 (`pruning/strategies/dart.py`)                  |
================================================================================
文件功能:
本文件实现了 DART (Duplication-Aware Related Token selection) 策略。

核心思想:
基于信息重复度（Duplication）而非单纯的重要性（Importance）进行剪枝。
1. **Pivot Selection**: 基于 Key 向量 L1 范数选择少量最具代表性的视觉和文本枢轴。
2. **Related Token Selection**: 围绕每个枢轴，收集与其特征重复度最低（即 Cosine 距离最大）的 Token。

来源论文:
"Stop Looking for Important Tokens in Multimodal Language Models:
Duplication Matters More (arXiv:2502.11494)"
"""
import torch
import torch.nn.functional as F
from typing import Dict, Any

# 从同级 utils 导入
from .utils import _extract_and_validate_vision_token_info


def dart_pruning(context: Dict[str, Any], **kwargs) -> torch.Tensor:
    """
    DART 剪枝策略函数。

    需要 `feature_map` 和 `pre_rope_k` (或 `post_rope_k`) 在 context 中。

    Args from kwargs:
        ratio (float): 要剪枝掉的视觉Token的比例。(必须提供)
        pivot_image_token (int or float): 要选择的视觉枢轴Token数量或比例。(必须提供)
        pivot_text_token (int or float): 要选择的文本枢轴Token数量或比例。(必须提供)
        use_post_rope (bool): 是否使用RoPE之后的Key向量。(必须提供)
    """
    # 1. 参数解析
    try:
        ratio = kwargs["ratio"]
        pivot_image_token = kwargs["pivot_image_token"]
        pivot_text_token = kwargs["pivot_text_token"]
        use_post_rope = kwargs["use_post_rope"]
    except KeyError as e:
        raise ValueError(f"Missing required parameter in kwargs for DART pruning: {e}")

    input_ids = context.get("input_ids")
    feature_map = context.get("feature_map")  # [B, Seq, Dim]
    k_tensor_key = "post_rope_k" if use_post_rope else "pre_rope_k"
    k_tensor = context.get(k_tensor_key)  # [B, H_kv, Seq, Dim_head]

    if input_ids is None:
        raise ValueError("Context 中缺少 'input_ids'。")
    if feature_map is None:
        raise ValueError("DART requires 'feature_map' in context.")
    if k_tensor is None:
        raise ValueError(f"DART requires '{k_tensor_key}' in context.")

    bsz, seq_len = input_ids.shape
    device = input_ids.device

    assert bsz == 1, "DART pruning currently asserts batch_size=1"

    batch_keep_masks = []
    for i in range(bsz):
        input_ids_single = input_ids[i]
        features_single = feature_map[i]  # [Seq, Dim]
        # k_tensor shape [B, H_kv, Seq, Dim_head] -> [Seq, H_kv * Dim_head]
        k_single = k_tensor[i].permute(1, 0, 2).reshape(seq_len, -1)

        # 3. 定位与分组
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

        # 4. 处理浮点数参数
        num_img_pivots_req = pivot_image_token
        num_txt_pivots_req = pivot_text_token
        if isinstance(num_img_pivots_req, float):
            num_img_pivots_req = int(num_img_pivots_req * num_vision_tokens)
        if isinstance(num_txt_pivots_req, float):
            num_txt_pivots_req = int(num_txt_pivots_req * len(non_vision_indices))

        # 5. 阶段一：选择决定性的枢轴Token (Based on Key L1 Norm)

        # 文本枢轴通常来自问题部分，这里简化为所有非视觉 token
        text_indices_for_pivot = non_vision_indices.to(device)

        k_visual = k_single[vision_indices]
        k_visual_l1_norm = torch.norm(k_visual, p=1, dim=-1)

        # 确保枢轴数量不超限
        num_img_pivots = min(num_img_pivots_req, num_vision_tokens, num_to_keep)
        img_pivot_indices = torch.tensor([], dtype=torch.long, device=device)
        top_img_pivot_local_indices = torch.tensor([], dtype=torch.long, device=device)
        if num_img_pivots > 0:
            _, top_img_pivot_local_indices = torch.topk(
                k_visual_l1_norm, k=num_img_pivots
            )
            img_pivot_indices = vision_indices[top_img_pivot_local_indices]

        txt_pivot_indices = torch.tensor([], dtype=torch.long, device=device)
        if len(text_indices_for_pivot) > 0:
            k_text = k_single[text_indices_for_pivot]
            k_text_l1_norm = torch.norm(k_text, p=1, dim=-1)
            num_txt_pivots = min(num_txt_pivots_req, len(text_indices_for_pivot))
            if num_txt_pivots > 0:
                _, top_txt_pivot_local_indices = torch.topk(
                    k_text_l1_norm, k=num_txt_pivots
                )
                txt_pivot_indices = text_indices_for_pivot[top_txt_pivot_local_indices]

        pivot_indices = torch.cat([img_pivot_indices, txt_pivot_indices])

        # 初始保留的视觉 token 是视觉枢轴
        kept_local_indices = set(top_img_pivot_local_indices.cpu().tolist())

        # 6. 阶段二：收集低重复度的相关Token
        total_pivots = len(pivot_indices)
        if total_pivots > 0 and len(kept_local_indices) < num_to_keep:
            num_remaining_to_keep = max(0, num_to_keep - len(kept_local_indices))
            # 每个 pivot 大致需要收集多少个
            token_topk_per_pivot = (
                num_remaining_to_keep + total_pivots - 1
            ) // total_pivots

            visual_features = features_single[vision_indices]
            visual_features_norm = F.normalize(visual_features, p=2, dim=1)

            for pivot_global_idx in pivot_indices:
                if len(kept_local_indices) >= num_to_keep:
                    break

                pivot_feature_norm = F.normalize(
                    features_single[pivot_global_idx].unsqueeze(0), p=2, dim=1
                )
                # 计算余弦相似度 [1, Dim] * [N_vision, Dim]^T -> [1, N_vision]
                cos_sim = torch.mm(
                    pivot_feature_norm, visual_features_norm.t()
                ).squeeze(
                    0
                )  # [N_vision]

                # 排除已选的视觉 token
                if kept_local_indices:
                    kept_indices_tensor = torch.tensor(
                        list(kept_local_indices), device=device, dtype=torch.long
                    )
                    # 将已选位置的相似度设为无穷大，这样 topk(largest=False) 就不会选它们
                    cos_sim.scatter_(0, kept_indices_tensor, float("inf"))

                # 需要收集的数量
                num_to_gather = min(
                    token_topk_per_pivot, num_to_keep - len(kept_local_indices)
                )
                # 实际可供选择的数量
                num_available = (cos_sim != float("inf")).sum().item()
                num_to_gather = min(num_to_gather, num_available)

                if num_to_gather <= 0:
                    continue

                # 找到最不相似的 topk (即重复度最低)
                # largest=False means select smallest similarity
                _, least_similar_local_indices = torch.topk(
                    cos_sim, k=num_to_gather, largest=False
                )

                kept_local_indices.update(least_similar_local_indices.cpu().tolist())

        # 7. 构建掩码
        final_kept_vision_indices = torch.tensor([], dtype=torch.long, device=device)
        if kept_local_indices:
            kept_local_indices_tensor = torch.tensor(
                list(kept_local_indices), device=device, dtype=torch.long
            )
            # 确保索引在 vision_indices 范围内
            valid_local_indices = kept_local_indices_tensor[
                kept_local_indices_tensor < len(vision_indices)
            ]
            if len(valid_local_indices) > 0:
                final_kept_vision_indices = vision_indices[valid_local_indices]

        final_kept_indices = torch.cat(
            [non_vision_indices.to(device), final_kept_vision_indices]
        )

        keep_mask = torch.zeros_like(input_ids_single, dtype=torch.bool)
        if final_kept_indices.numel() > 0:
            keep_mask[final_kept_indices] = True
        batch_keep_masks.append(keep_mask)

    return torch.stack(batch_keep_masks, dim=0)