# -*- coding: utf-8 -*-
"""
================================================================================
|         基于注意力的剪枝策略 (`pruning/strategies/attention_based.py`) v1.1       |
================================================================================
文件功能:
本文件包含依赖模型内部注意力机制计算分数的剪枝策略函数。

v1.1 更新:
- [Bug修复] 修复了 dominant_token_pruning 中因在 GPU Tensor 上调用 np.cumsum
  导致的 TypeError。将 np.cumsum 替换为 torch.cumsum 并添加必要的类型/设备转换。
"""
import torch
import math
from typing import Dict, Any, Tuple, List
import numpy as np  # 保持导入 numpy 以备其他地方使用

# 从同级 utils 导入
from .utils import (
    _extract_and_validate_vision_token_info,
    _recompute_attention_maps_for_all_images,
    identify_model_architecture,
)

# 从上级目录导入 Token ID
from ..pruning_strategies import PRUNABLE_TOKEN_IDS, IM_END_ID

# --- 基于注意力的策略函数 ---


def special_token_based_attention_pruning(
    context: Dict[str, Any], **kwargs
) -> torch.Tensor:
    """
    基于特定Token的Query与视觉Token的Key之间的注意力分数进行剪枝。

    此方法不依赖完整的注意力图，而是直接计算Q-K点积作为重要性分数。
    'last_text'策略的灵感来源于 "An Image is Worth 1/2 Tokens After
    Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models"。

    需要 `pre_rope_q`, `pre_rope_k` 或 `post_rope_q`, `post_rope_k` 在 context 中。

    Args from kwargs:
        ratio (float): 剪枝比例。(必须提供)
        use_post_rope (bool): 是否使用RoPE之后的Q/K向量。(必须提供)
        query_source (dict): Query向量的配置策略。(必须提供)
          - strategy (str): 'last_text', 'last_word_before_final_im_end', 'special_token'
          - token_id (int, optional): 如果 strategy='special_token', 指定特殊 token ID。
          - scoping (str, optional): 如果 strategy='special_token', 指定范围 ('global_average', 'global_last')。
    """
    # 1. 参数解析
    try:
        ratio = kwargs["ratio"]
        use_post_rope = kwargs["use_post_rope"]
        query_source = kwargs["query_source"]
        strategy = query_source["strategy"]
    except KeyError as e:
        raise ValueError(
            f"Missing required parameter in kwargs for attention pruning: {e}"
        )

    input_ids = context.get("input_ids")
    if input_ids is None:
        raise ValueError("Context 中缺少 'input_ids'。")

    q_tensor_key = "post_rope_q" if use_post_rope else "pre_rope_q"
    k_tensor_key = "post_rope_k" if use_post_rope else "pre_rope_k"

    q = context.get(q_tensor_key)
    k = context.get(k_tensor_key)

    if q is None or k is None:
        raise ValueError(
            f"Context 中缺少 '{q_tensor_key}' 或 '{k_tensor_key}'。请检查 'needs' 配置。"
        )

    bsz, seq_len = input_ids.shape
    device = input_ids.device
    assert (
        bsz == 1
    ), "Special token based attention pruning currently asserts batch_size=1"

    batch_keep_masks = []
    for i in range(bsz):  # 虽然断言了 bsz=1，但保持循环结构
        input_ids_single = input_ids[i]
        # q/k shape: [B, H, Seq, Dim] -> [H, Seq, Dim]
        q_single = q[i]
        k_single = k[i]

        # 2. 定位与分组
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

        # 3. 获取Query向量
        query_indices = []
        if strategy == "last_text":
            # 找到最后一个非视觉 token
            if len(non_vision_indices) > 0:
                query_indices.append(non_vision_indices[-1])
        elif strategy == "last_word_before_final_im_end":
            if identify_model_architecture(context) == "llava":
                raise ValueError("last_word_before_final_im_end 不支持 LLaVA 模型。")
            im_end_indices = torch.where(input_ids_single == IM_END_ID)[0]
            if len(im_end_indices) > 0:
                last_im_end_idx = im_end_indices[-1]
                # 确保前面有 token
                if last_im_end_idx > 0:
                    query_indices.append(last_im_end_idx - 1)
        elif strategy == "special_token":
            if identify_model_architecture(context) == "llava":
                raise ValueError("special_token 不支持 LLaVA 模型。")
            scoping = query_source.get("scoping")  # 使用 .get()
            token_id = query_source.get("token_id")
            if token_id is None or scoping is None:
                raise ValueError(
                    "'special_token' strategy requires 'token_id' and 'scoping'."
                )

            found_indices = torch.where(input_ids_single == token_id)[0]
            if len(found_indices) > 0:
                if scoping == "global_average":
                    query_indices.extend(found_indices.tolist())
                elif scoping == "global_last":
                    query_indices.append(found_indices[-1])
                # 可以添加 'first' 等其他 scope
            else:
                print(f"警告: 在序列中未找到指定的 special_token ID: {token_id}")

        if not query_indices:
            print(
                f"警告: 未能根据策略 '{strategy}' 找到有效的 query token 索引，将保留所有视觉 token。"
            )
            batch_keep_masks.append(torch.ones_like(input_ids_single, dtype=torch.bool))
            continue

        # 转换为 tensor 索引
        query_indices_tensor = torch.tensor(
            query_indices, device=device, dtype=torch.long
        )
        # 获取 Q 向量并取平均 [H, Seq, Dim] -> [H, N_query, Dim] -> [H, 1, Dim]
        q_query = q_single[:, query_indices_tensor, :].mean(dim=1, keepdim=True)

        # 4. 计算注意力分数并剪枝
        # k_single shape: [H_kv, Seq, Dim]
        k_visual = k_single[:, vision_indices, :]  # [H_kv, N_vision, Dim]
        head_dim = q_query.shape[-1]
        num_q_heads = q_query.shape[0]
        num_kv_heads = k_visual.shape[0]

        if num_q_heads != num_kv_heads:  # GQA/MQA handling
            if num_q_heads % num_kv_heads != 0:
                raise ValueError(
                    "num_q_heads must be divisible by num_kv_heads for GQA/MQA."
                )
            num_key_value_groups = num_q_heads // num_kv_heads
            k_visual = k_visual.repeat_interleave(
                num_key_value_groups, dim=0
            )  # [H_q, N_vision, Dim]

        # 计算 QK^T [H, 1, Dim] * [H, N_vision, Dim]^T -> [H, 1, N_vision]
        attn_scores = torch.einsum("hid,hjd->hij", q_query, k_visual)

        scaled_scores = attn_scores / math.sqrt(head_dim)  # [H, 1, N_vision]
        # Softmax 在最后一个维度 (视觉 token 维度)
        softmax_scores = torch.softmax(scaled_scores, dim=-1)  # [H, 1, N_vision]

        # 对 Head 维度求平均得到最终分数 [1, N_vision] -> [N_vision]
        final_scores = softmax_scores.mean(dim=0).squeeze(0)

        # 获取 top K
        kept_vision_indices = torch.tensor([], dtype=torch.long, device=device)
        if num_to_keep > 0:
            k = min(num_to_keep, num_vision_tokens)
            _, kept_indices_local = torch.topk(final_scores, k=k)
            kept_vision_indices = vision_indices[kept_indices_local]

        # 5. 构建掩码
        final_kept_indices = torch.cat(
            [non_vision_indices.to(device), kept_vision_indices]
        )
        keep_mask = torch.zeros_like(input_ids_single, dtype=torch.bool)
        if final_kept_indices.numel() > 0:
            keep_mask[final_kept_indices] = True
        batch_keep_masks.append(keep_mask)

    return torch.stack(batch_keep_masks, dim=0)
