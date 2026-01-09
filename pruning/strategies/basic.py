# -*- coding: utf-8 -*-
"""
================================================================================
|             基础剪枝策略 (`pruning/strategies/basic.py`)                   |
================================================================================
文件功能:
本文件包含一些基础的、不依赖复杂上下文信息的启发式剪枝策略函数。
这些策略通常作为性能基线或简单的加速手段。
"""
import torch
from typing import Dict, Any, Tuple
import numpy as np  # 导入 numpy

# 从同级 utils 导入
from .utils import _extract_and_validate_vision_token_info

# 从上级目录导入 Token ID
from ..pruning_strategies import PRUNABLE_TOKEN_IDS

# --- 基础策略函数 ---


def baseline_pruning(context: Dict[str, Any], **kwargs) -> torch.Tensor:
    """
    [新增] 基准线（Baseline）策略，不执行任何剪枝。

    此策略仅用于评测框架中作为对照组，它总是返回一个全为 True 的掩码，
    表示保留所有 Token。

    Args (来自 kwargs):
        (无)

    Returns:
        torch.Tensor:
            一个形状为 `[B, seq_len]` 的全 `True` 布尔型掩码。
    """
    input_ids = context.get("input_ids")
    if input_ids is None:
        raise ValueError("Context 中缺少 'input_ids'。")

    # 返回一个与 input_ids 形状相同、设备相同、全为 True 的掩码
    return torch.ones_like(input_ids, dtype=torch.bool)


def override_pruning(context: Dict[str, Any], **kwargs) -> torch.Tensor:
    """
    [修改] 按预设的视觉Token掩码进行剪枝。

    此策略接收一个只针对视觉部分的掩码，并将其应用到正确的Token位置上，
    同时自动保留所有非视觉Token。

    Args from kwargs:
        mask (torch.Tensor): 一个只包含视觉Token部分的、预先计算好的
                             布尔型 `keep_mask` 张量。(必须提供)
    """
    partial_mask = kwargs.get("mask")
    if partial_mask is None:
        raise ValueError("`override_pruning` 策略需要在 params 中提供 'mask' 参数。")
    if not isinstance(partial_mask, torch.Tensor) or partial_mask.dtype != torch.bool:
        raise TypeError("提供的 'mask' 必须是一个布尔型 PyTorch 张量。")

    input_ids = context.get("input_ids")  # 使用 get
    if input_ids is None:
        raise ValueError("Context 中缺少 'input_ids'。")

    assert input_ids.shape[0] == 1, "override_pruning 当前断言 batch_size=1"

    # 1. 获取视觉Token的准确位置
    # 确保 partial_mask 和 vision_indices 在同一设备
    device = input_ids.device
    partial_mask = partial_mask.to(device)
    vision_indices, _, prunable_mask, _ = _extract_and_validate_vision_token_info(
        context
    )

    # 2. 校验传入的掩码长度是否与视觉Token数量一致
    if len(partial_mask) != len(vision_indices):
        # 尝试处理可能的维度问题
        if partial_mask.dim() > 1:
            partial_mask = partial_mask.flatten()
        if len(partial_mask) != len(vision_indices):
            raise ValueError(
                f"提供的部分掩码长度 ({len(partial_mask)}) 与在输入中找到的视觉Token数量 ({len(vision_indices)}) 不匹配。"
                f" Mask shape: {partial_mask.shape}"
            )

    # 3. 构建完整的 keep_mask
    full_keep_mask = ~prunable_mask  # 默认保留所有非视觉Token
    # 确保 vision_indices 非空再索引赋值
    if len(vision_indices) > 0:
        full_keep_mask[vision_indices] = partial_mask  # 将部分掩码应用到视觉Token的位置

    return full_keep_mask.unsqueeze(0)  # 增加batch维度以符合标准输出


def random_pruning(context: Dict[str, Any], **kwargs) -> torch.Tensor:
    """
    随机剪枝指定比例的视觉Token。

    Args from kwargs:
        ratio (float): 要剪枝掉的视觉Token的比例，取值范围 [0.0, 1.0]。
                       (必须提供)
    """
    try:
        ratio = kwargs["ratio"]
    except KeyError:
        raise ValueError("`random_pruning` strategy requires 'ratio' in kwargs.")

    input_ids = context.get("input_ids")
    if input_ids is None:
        raise ValueError("Context 中缺少 'input_ids'。")

    bsz, seq_len = input_ids.shape
    device = input_ids.device

    assert bsz == 1, "Random pruning currently asserts batch_size=1"

    batch_keep_masks = []
    for i in range(bsz):
        input_ids_single = input_ids[i]

        # 直接使用工具函数获取索引
        vision_indices, non_vision_indices, _, _ = (
            _extract_and_validate_vision_token_info(context)
        )

        num_vision_tokens = len(vision_indices)
        if num_vision_tokens == 0:
            batch_keep_masks.append(torch.ones_like(input_ids_single, dtype=torch.bool))
            continue

        num_to_keep = int(round(num_vision_tokens * (1 - ratio)))
        # 确保至少保留一个（如果原本有的话），除非 ratio=1.0
        if ratio < 1.0 and num_to_keep == 0 and num_vision_tokens > 0:
            num_to_keep = 1
        # 如果 ratio=1.0 且 num_to_keep=0，则不保留

        kept_vision_indices = torch.tensor([], dtype=torch.long, device=device)
        if num_to_keep > 0:
            shuffled_indices = vision_indices[
                torch.randperm(num_vision_tokens, device=device)
            ]
            kept_vision_indices = shuffled_indices[:num_to_keep]

        # 确保 non_vision_indices 和 kept_vision_indices 都在同一设备
        final_kept_indices = torch.cat(
            [non_vision_indices.to(device), kept_vision_indices]
        )

        keep_mask = torch.zeros_like(input_ids_single, dtype=torch.bool)
        # 确保索引有效再赋值
        if len(final_kept_indices) > 0:
            keep_mask[final_kept_indices] = True
        batch_keep_masks.append(keep_mask)

    return torch.stack(batch_keep_masks, dim=0)

