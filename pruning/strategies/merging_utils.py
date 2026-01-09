# -*- coding: utf-8 -*-
"""
================================================================================
|       合并策略辅助工具模块 (`pruning/strategies/merging_utils.py`) v1.1         |
================================================================================
文件功能:
本文件是 Token 合并策略的专属工具库，提供了计算和处理合并所需的基础函数。

v1.1 更新:
- [适配] `get_dialogue_masks` 增加对 `llava_ov` (LLaVA-OneVision) 的支持，
  将其视为通用 LLaVA 类型处理（暂不解析特定角色掩码）。
"""

import torch
from typing import Any, Dict, Optional, Tuple, List
import bisect  # 用于二分查找

# --- 定义特殊 Token ID ---
IM_START = 151644
IM_END = 151645
ROLE_USER = 872
ROLE_ASSIST = 77091
NL = 198

IMG_START = 151652
IMG_ELEM = 151655
IMG_END = 151653

# --- 预构建起始序列 Tensor (放在 CPU 上) ---
USER_START_SEQ = torch.tensor([IM_START, ROLE_USER, NL], device="cpu")
ASSIST_START_SEQ = torch.tensor([IM_START, ROLE_ASSIST, NL], device="cpu")


def find_sequence_indices(tensor_1d: torch.Tensor, sequence: torch.Tensor) -> List[int]:
    """
    在 1D 张量中查找一个子序列的所有起始索引。
    计算在 CPU 上执行以使用 unfold 和 list anipulation。
    """
    assert tensor_1d.device.type == "cpu", "输入 tensor_1d 必须在 CPU"
    assert sequence.device.type == "cpu", "输入 sequence 必须在 CPU"
    seq_len = sequence.shape[0]
    tensor_len = tensor_1d.shape[0]
    if tensor_len < seq_len:
        return []
    windows = tensor_1d.unfold(0, seq_len, 1)
    matches = torch.all(windows == sequence, dim=1)
    indices = torch.nonzero(matches, as_tuple=False).squeeze(-1).tolist()
    return indices


def find_next_greater_index(sorted_indices: List[int], value: int) -> int:
    """
    在已排序的索引列表中，查找第一个大于给定值的索引。
    """
    insert_point = bisect.bisect_right(sorted_indices, value)
    if insert_point < len(sorted_indices):
        return sorted_indices[insert_point]
    return -1


def get_dialogue_masks(
    context: Dict[str, Any],
) -> Tuple[
    Optional[torch.Tensor],
    List[torch.Tensor],
    Optional[torch.Tensor],
    List[int],
    List[bool],
]:
    """
    [v1.3 修改] 分析对话结构。支持利用 cu_seqlens_full 并结合空间池化倍率对连续视觉 Token 进行精确的物理单元（帧/图）细分。
    """
    # 1. 基础数据准备
    input_ids = context.get("input_ids")
    if input_ids is None:
        raise ValueError("Context 中缺少必要的 'input_ids'。")

    assert (
        input_ids.shape[0] == 1
    ), f"目前仅支持 batch size 为 1，但得到的是 {input_ids.shape[0]}"

    device = input_ids.device
    input_ids_1d = input_ids.squeeze(0)
    seq_len = input_ids_1d.shape[0]

    # 导入辅助工具
    from .utils import (
        _extract_and_validate_vision_token_info,
        identify_model_architecture,
    )

    model_type = identify_model_architecture(context)
    _, _, prunable_mask, _ = _extract_and_validate_vision_token_info(context)

    # --- [核心修改 1] 换算物理单元长度 (考虑池化和模型架构差异) ---
    cu_seqlens = context.get("cu_seqlens_full")
    merged_unit_lengths = []
    
    if cu_seqlens is not None:
        spatial_merge_size = context.get("spatial_merge_size", 2)
        merge_unit = spatial_merge_size ** 2
        
        # cu_seqlens 为累积长度 [0, L1, L1+L2, ...], diff 得到每段细粒度长度
        fine_lengths = torch.diff(cu_seqlens).cpu().tolist()
        for flen in fine_lengths:
            # 逻辑参考 utils.py：LLaVA-OV 每个单元含 1 个 CLS Token，池化时被丢弃
            actual_fine_len = flen - 1 if model_type == "llava_ov" else flen
            mlen = actual_fine_len // merge_unit
            if mlen > 0:
                merged_unit_lengths.append(mlen)
        
        # 严格验证：计算出的物理单元总长必须等于 input_ids 中发现的视觉 Token 总数
        total_vision_tokens_found = prunable_mask.sum().item()
        if sum(merged_unit_lengths) != total_vision_tokens_found:
            raise ValueError(
                f"[Consistency Error] 计算出的物理单元总长 ({sum(merged_unit_lengths)}) "
                f"与序列中实际发现的视觉 Token 数 ({total_vision_tokens_found}) 不匹配！"
                f"模型类型: {model_type}, 池化倍率: {merge_unit}"
            )

    unit_ptr = 0 # 追踪消费到哪个物理单元了

    # 2. 基于掩码变化进行初步分段 (划分文本段与视觉岛屿)
    changes = (
        torch.nonzero(prunable_mask[1:] != prunable_mask[:-1], as_tuple=False).flatten()
        + 1
    )
    boundaries = torch.cat(
        [
            torch.tensor([0], device=device),
            changes,
            torch.tensor([seq_len], device=device),
        ]
    )

    sizes_list = []
    is_vision_token_list = []
    image_masks_list = []

    # 3. 遍历初步分段，执行物理单元级别的细分
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i].item(), boundaries[i + 1].item()
        segment_size = end - start
        if segment_size <= 0:
            continue

        is_vis = bool(prunable_mask[start].item())

        if is_vis and merged_unit_lengths:
            # --- [核心修改 2] 细分视觉岛屿：贪心消费该段包含的物理单元 ---
            consumed_in_segment = 0
            while consumed_in_segment < segment_size:
                if unit_ptr >= len(merged_unit_lengths):
                    raise RuntimeError("检测到视觉 Token 段，但 cu_seqlens 定义的物理单元已提前耗尽。")
                
                curr_unit_len = merged_unit_lengths[unit_ptr]
                sizes_list.append(curr_unit_len)
                is_vision_token_list.append(True)
                
                # 为该独立物理单元（帧/图）创建专用掩码
                u_start = start + consumed_in_segment
                u_end = u_start + curr_unit_len
                img_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
                img_mask[0, u_start:u_end] = True
                image_masks_list.append(img_mask)
                
                consumed_in_segment += curr_unit_len
                unit_ptr += 1
            
            # 内部校验：确保该视觉岛屿被完整且精确地拆分
            if consumed_in_segment != segment_size:
                raise ValueError(
                    f"视觉岛屿长度 ({segment_size}) 无法由 cu_seqlens 中的单元精确组成 (消费了 {consumed_in_segment})。"
                    f"请检查模型是否在图像块之间插入了未识别的占位 Token。"
                )
        else:
            # 文本片段或普通视觉片段 (无 cu_seqlens 时的降级处理)
            sizes_list.append(segment_size)
            is_vision_token_list.append(is_vis)
            if is_vis:
                img_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
                img_mask[0, start:end] = True
                image_masks_list.append(img_mask)

    # 4. 最终对齐与完整性检查
    if sum(sizes_list) != seq_len:
        raise ValueError(f"分段长度总和 ({sum(sizes_list)}) 与序列总长度 ({seq_len}) 不符。")
    if merged_unit_lengths and unit_ptr != len(merged_unit_lengths):
        raise ValueError(f"cu_seqlens 中的物理单元未被完全消费 (剩余 {len(merged_unit_lengths) - unit_ptr} 个)。")

    # 5. 模型特定的角色掩码处理 (逻辑保持不变)
    user_text_mask, assistant_text_mask = None, None
    if model_type not in ["llava", "llava_ov"] and "qwen" in model_type:
        seq_cpu = input_ids_1d.cpu()
        user_text_mask_cpu = torch.zeros(seq_len, dtype=torch.bool)
        assistant_text_mask_cpu = torch.zeros(seq_len, dtype=torch.bool)
        im_end_indices = torch.nonzero(seq_cpu == IM_END, as_tuple=False).squeeze(-1).tolist()
        user_start_indices = find_sequence_indices(seq_cpu, USER_START_SEQ)
        assist_start_indices = find_sequence_indices(seq_cpu, ASSIST_START_SEQ)
        def get_next_im_end_idx(pos: int) -> int:
            idx = bisect.bisect_right(im_end_indices, pos)
            return im_end_indices[idx] if idx < len(im_end_indices) else seq_len
        for s_idx in user_start_indices:
            user_text_mask_cpu[s_idx + len(USER_START_SEQ) : get_next_im_end_idx(s_idx)] = True
        for s_idx in assist_start_indices:
            assistant_text_mask_cpu[s_idx + len(ASSIST_START_SEQ) : get_next_im_end_idx(s_idx)] = True
        user_text_mask = (user_text_mask_cpu.to(device) & (~prunable_mask)).unsqueeze(0)
        assistant_text_mask = (assistant_text_mask_cpu.to(device) & (~prunable_mask)).unsqueeze(0)

    return (
        user_text_mask,
        image_masks_list,
        assistant_text_mask,
        sizes_list,
        is_vision_token_list,
    )

def create_fake_mask_from_merging_results(
    merge_weight_list: List[torch.Tensor],
    sizes_list: List[int],  # CPU 列表
    is_vision_list: List[bool],  # CPU 列表
    device: torch.device,
) -> torch.Tensor:
    """
    [v1.0 新增] 根据合并策略的输出（合并矩阵列表和片段信息）构建一个
    与 PruningCache 兼容的“伪掩码”(fake_mask)。

    此掩码的形状与*原始*序列长度 N 相同。
    - 文本片段 (is_vision=False) 对应位置全为 True。
    - 视觉片段 (is_vision=True) 对应位置包含 M_i 个 True 和 (N_i - M_i) 个 False。

    Args:
        merge_weight_list (List[torch.Tensor]):
            一个 Python 列表，包含*每个*视觉片段的合并变换矩阵 T_i。
            每个张量 T_i 的形状为 [M_i, N_i]。
        sizes_list (List[int]):
            从 get_dialogue_masks 获取的 Python 列表，包含所有片段的长度。
        is_vision_list (List[bool]):
            从 get_dialogue_masks 获取的 Python 列表，标记每个片段是否为视觉片段。
        device (torch.device):
            希望最终 fake_mask 所在的设备 (例如 'cuda:0')。

    Returns:
        torch.Tensor:
            一个 1D GPU 布尔张量（形状 [N]），代表合并后的“伪掩码”。
    """
    if len(sizes_list) != len(is_vision_list):
        raise ValueError(
            f"sizes_list (长度 {len(sizes_list)}) 和 is_vision_list (长度 {len(is_vision_list)}) 必须具有相同的长度。"
        )

    fake_mask_segments = []
    merge_weight_iter = iter(merge_weight_list)

    for segment_size, is_vision in zip(sizes_list, is_vision_list):
        if is_vision:
            # --- 视觉片段 ---
            try:
                T_i = next(merge_weight_iter)
            except StopIteration:
                raise RuntimeError(
                    "合并策略返回的 merge_weight_list 数量与 is_vision_list 中的视觉片段数量不匹配。"
                )

            N_i = T_i.shape[1]  # 原始 Token 数
            M_i = T_i.shape[0]  # 合并后 Token 数

            # 验证 `sizes_list` 中的长度是否与合并矩阵的 N_i 匹配
            if N_i != segment_size:
                raise ValueError(
                    f"sizes_list 中的视觉片段长度 ({segment_size}) 与 "
                    f"对应的合并矩阵 T_i 的原始维度 N_i ({N_i}) 不匹配。"
                )

            if M_i > N_i:
                raise ValueError(
                    f"合并矩阵 T_i 的形状无效：合并后的 Token 数 ({M_i}) 不能大于原始 Token 数 ({N_i})。"
                )

            # 创建 M_i 个 True (保留)
            kept_mask_part = torch.ones(M_i, dtype=torch.bool, device=device)
            # 创建 N_i - M_i 个 False (被合并)
            pruned_mask_part = torch.zeros(N_i - M_i, dtype=torch.bool, device=device)

            # 拼接成 N_i 长度的片段掩码
            segment_mask = torch.cat([kept_mask_part, pruned_mask_part])
            fake_mask_segments.append(segment_mask)
        else:
            # --- 文本/特殊符号片段 ---
            L_j = segment_size
            segment_mask = torch.ones(L_j, dtype=torch.bool, device=device)
            fake_mask_segments.append(segment_mask)

    # 检查是否所有合并矩阵都已使用
    try:
        next(merge_weight_iter)
        # 如果还能迭代，说明 merge_weight_list 太长了
        # print(
        #     "Warning: create_fake_mask: merge_weight_list 包含的矩阵多于 is_vision_list 中的视觉片段数量。"
        # )
    except StopIteration:
        pass  # 正常情况

    # 拼接所有片段，创建最终的 [N] 形状掩码
    if not fake_mask_segments:
        return torch.tensor([], dtype=torch.bool, device=device)

    fake_mask = torch.cat(fake_mask_segments, dim=0)
    return fake_mask