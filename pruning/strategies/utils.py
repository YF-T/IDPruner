# -*- coding: utf-8 -*-
"""
================================================================================
|       剪枝策略辅助工具函数 (`pruning/strategies/utils.py`) v4.4 (Strict Check) |
================================================================================
文件功能:
本文件包含支持各种剪枝策略计算的辅助函数。

v4.4 更新:
- [Strict] 对 LLaVA-OV 启用严格的 Token 数量校验，不再允许不匹配（CLS Token 仅在 ViT 内部，不影响 input_ids）。
"""
import torch
import math
from typing import Dict, Any, List, Set, Tuple, Optional
import torch.nn.functional as F

# === modify start: update supported types ===
SUPPORTED_MODEL_TYPES = {"llava", "qwen2_5_vl", "llava_ov"}
# === modify end ===


def _get_config_attr(config: Any, attr_name: str, default: Any = None) -> Any:
    """兼容 Config 对象和 Dict 的属性获取辅助函数"""
    if isinstance(config, dict):
        return config.get(attr_name, default)
    return getattr(config, attr_name, default)


def identify_model_architecture(context: Dict[str, Any]) -> str:
    """
    根据 context 中的 model_config 识别模型架构类型。
    """
    config = context.get("model_config")
    if not config:
        # 严格报错
        raise ValueError("Context 中缺少 'model_config'，无法识别模型架构。")

    model_type = str(_get_config_attr(config, "model_type")).lower()

    # === modify start: identify llava_ov ===
    # LLaVA-OneVision 使用 'rice_vit'，这是与传统 LLaVA 最显著的区别
    vision_config = _get_config_attr(config, "vision_config")
    if vision_config:
        vision_model_type = str(_get_config_attr(vision_config, "model_type", "")).lower()
        if "rice" in vision_model_type:
            return "llava_ov"
    
    if "llavaonevision" in model_type:
        return "llava_ov"
    # === modify end ===

    if "llava" in model_type:
        return "llava"
    if "qwen2_5_vl" in model_type:
        return "qwen2_5_vl"

    raise ValueError(f"不支持的模型类型: '{model_type}'。")


def get_model_specific_vision_token_ids(context: Dict[str, Any]) -> Set[int]:
    """
    获取当前模型特定的所有视觉相关 Token ID。
    """
    model_type = identify_model_architecture(context)
    config = context.get("model_config")
    token_ids = set()

    if model_type == "llava":
        img_id = _get_config_attr(config, "image_token_index")
        if img_id is not None:
            token_ids.add(img_id)
        img_id_alt = _get_config_attr(config, "image_token_id")
        if img_id_alt is not None:
            token_ids.add(img_id_alt)

    # === modify start: allow llava_ov to use qwen-like ids ===
    elif "qwen" in model_type or model_type == "llava_ov":
        keys_to_fetch = [
            "image_token_id",
            "video_token_id",
            # "vision_start_token_id",
            # "vision_end_token_id",
            "vision_token_id",
        ]
        for key in keys_to_fetch:
            val = _get_config_attr(config, key)
            if val is not None:
                token_ids.add(val)
    # === modify end ===

    if not token_ids:
        # 尝试通用兜底
        img_id = _get_config_attr(config, "image_token_id")
        if img_id is not None:
            token_ids.add(img_id)

    if not token_ids:
        raise ValueError(f"无法从配置中提取视觉 Token ID (Model: {model_type})。")

    return token_ids


def _extract_and_validate_vision_token_info(
    context: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    通用辅助函数：定位视觉Token，并推断每个图像块实际包含的视觉Token数量。
    """
    input_ids = context.get("input_ids")
    if input_ids is None:
        raise ValueError("Context 中缺少必要的 'input_ids'。")

    bsz = input_ids.shape[0]
    device = input_ids.device
    if bsz != 1:
        raise ValueError(f"此辅助函数仅支持 batch_size=1，收到 {bsz}")

    input_ids_single = input_ids.squeeze(0)

    model_type = identify_model_architecture(context)
    config = context.get("model_config")

    # 1. 构建 Mask
    target_token_ids = get_model_specific_vision_token_ids(context)
    prunable_mask = torch.zeros_like(input_ids_single, dtype=torch.bool, device=device)

    for tid in target_token_ids:
        prunable_mask |= input_ids_single == tid

    # 2. 提取索引
    vision_indices = torch.where(prunable_mask)[0]
    non_vision_indices = torch.where(~prunable_mask)[0]
    num_vision_tokens_found = len(vision_indices)

    # 3. 推断图像块
    num_merged_tokens_actual_per_image = []
    if num_vision_tokens_found > 0:
        vision_diffs = vision_indices[1:] - vision_indices[:-1]
        split_points = torch.where(vision_diffs > 1)[0] + 1
        image_token_blocks = torch.tensor_split(vision_indices, split_points.cpu())
        num_merged_tokens_actual_per_image = [
            len(block) for block in image_token_blocks
        ]

    # 4. 严格校验 (仅 Qwen 和 LLaVA-OV)
    # === modify start: enable strict check for llava_ov ===
    if "qwen" in model_type or model_type == "llava_ov":
        image_grid_thw = context.get("image_grid_thw")
        spatial_merge_size = 2  # 默认
        if config:
            vision_config = _get_config_attr(config, "vision_config")
            if vision_config:
                spatial_merge_size = _get_config_attr(
                    vision_config, "spatial_merge_size", 2
                )
        elif "spatial_merge_size" in context:
            spatial_merge_size = context["spatial_merge_size"]

        if image_grid_thw is not None:
            image_grid_thw_dev = image_grid_thw.to(device=device)
            num_merged_tokens_expected_per_image = (
                (
                    (image_grid_thw_dev[:, 1] // spatial_merge_size)
                    * (image_grid_thw_dev[:, 2] // spatial_merge_size)
                )
                .cpu()
                .tolist()
            )
            expected_total_tokens = sum(num_merged_tokens_expected_per_image)

            # [Fix]: 升级为 ValueError，因为 input_ids 不应包含 ViT 内部的 CLS
            if num_vision_tokens_found != expected_total_tokens:
                raise ValueError(
                    f"[Strict Check Failed] Token总数不匹配: input_ids中找到 {num_vision_tokens_found}, "
                    f"但根据 image_grid_thw 预期 {expected_total_tokens}。\n"
                    f"Grid: {image_grid_thw_dev.tolist()}, Merge Size: {spatial_merge_size}"
                )
            
            if (
                num_merged_tokens_actual_per_image
                != num_merged_tokens_expected_per_image
            ):
                raise ValueError(
                    f"[Strict Check Failed] 图像块结构不匹配: Input推断 {num_merged_tokens_actual_per_image} "
                    f"vs 元数据预期 {num_merged_tokens_expected_per_image}。"
                )
    # === modify end ===

    return (
        vision_indices,
        non_vision_indices,
        prunable_mask,
        num_merged_tokens_actual_per_image,
    )


def _recompute_attention_maps_for_all_images(
    q_tensor: torch.Tensor, k_tensor: torch.Tensor, context: Dict[str, Any]
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    [v4.2 接口清洁] 为每张图重计算注意力分数和特征 Key。

    Qwen / LLaVA-OV: 从 context['cu_seqlens_full'] 读取长度信息，处理 Packed Sequence。
    LLaVA: 直接处理 Batched Input [Num_Images, ...]。

    输出:
        Tuple[List[Score], List[Key]]:
        - final_scores_list: List[[1, N_merged]] (Attention Importance)
        - final_keys_list: List[[1, N_merged, HeadDim]] (Metric Features)
    """
    final_scores_list = []
    final_keys_list = []

    model_type = identify_model_architecture(context)
    head_dim = q_tensor.shape[-1]
    device = q_tensor.device

    # ==========================
    # 逻辑分支 A: LLaVA (Standard)
    # ==========================
    if model_type == "llava":
        # LLaVA 的 Q/K: [Num_Images, Heads, Seq, Dim] (adapter 已处理 transpose)
        # 如果 adapter 存的是 [B, Seq, Heads, Dim]，需先 Transpose
        # 假设 adapter 逻辑: query_states.view(bsz, tgt_len, heads, dim).transpose(1, 2) -> [B, Heads, Seq, Dim]

        if q_tensor.dim() != 4:
            raise ValueError(
                f"LLaVA Q Tensor 应为 4D [N_img, H, S, D]，收到 {q_tensor.shape}"
            )

        num_images = q_tensor.shape[0]

        for i in range(num_images):
            # [H, S, D]
            q_img = q_tensor[i]
            k_img = k_tensor[i]

            # 1. 提取 CLS (Index 0) 和 Patches (Index 1:)
            q_cls = q_img[:, 0:1, :]  # [H, 1, D]
            k_patches = k_img[:, 1:, :]  # [H, N-1, D]

            if k_patches.shape[1] == 0:
                raise ValueError(f"LLaVA Image {i}: 只有 CLS Token，没有 Patch。")

            # 2. Compute Attention
            attn_logits = torch.matmul(q_cls, k_patches.transpose(-1, -2)) / math.sqrt(
                head_dim
            )
            attn_weights = torch.softmax(attn_logits, dim=-1)

            # 3. Average over Heads -> [1, N_patch]
            score = attn_weights.mean(dim=0)  # [1, N_patch]
            final_scores_list.append(score)

            # 4. Key: [1, N_patch, D] (Mean over heads)
            # k_patches: [H, N_patch, D] -> mean(0) -> [N_patch, D] -> unsqueeze
            key_val = k_patches.mean(dim=0).unsqueeze(0)
            final_keys_list.append(key_val)

    # ==========================
    # 逻辑分支 B: Qwen2.5/3 和 LLaVA-OV (Packed Sequence)
    # ==========================
    # === modify start: include llava_ov in this branch ===
    elif "qwen" in model_type or model_type == "llava_ov":
    # === modify end ===
        # 获取配置
        spatial_merge_size = context.get("spatial_merge_size", 2)
        reverse_indices = context.get("reverse_indices")
        cu_seqlens = context.get("cu_seqlens_full")

        if cu_seqlens is None:
            raise ValueError(
                f"模型 ({model_type}) 需要 context['cu_seqlens_full']。"
            )

        # Qwen/OV Adapter 存的是: [1, Heads, Total_Seq, Dim]
        if q_tensor.dim() == 4:
            q_tensor = q_tensor.squeeze(0)
        if k_tensor.dim() == 4:
            k_tensor = k_tensor.squeeze(0)

        # 期望: [Heads, Total_Seq, Dim]
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu()
        total_len = lengths.sum().item()

        if q_tensor.shape[1] == total_len:  # [Heads, Seq, Dim]
            pass
        elif q_tensor.shape[0] == total_len:  # [Seq, Heads, Dim]
            q_tensor = q_tensor.transpose(0, 1)
            k_tensor = k_tensor.transpose(0, 1)
        else:
            raise ValueError(
                f"Q/K 长度 ({q_tensor.shape}) 与 cu_seqlens ({total_len}) 不匹配。"
            )

        # 切分
        try:
            q_splits = torch.split(q_tensor, lengths.tolist(), dim=1)
            k_splits = torch.split(k_tensor, lengths.tolist(), dim=1)
        except Exception as e:
            raise ValueError(f"Q/K Split 失败: {e}")

        qwen_merged_scores = []
        qwen_merged_keys = []

        for i, (q_slice, k_slice) in enumerate(zip(q_splits, k_splits)):
            if q_slice.numel() == 0:
                raise ValueError(f"Qwen Split {i} 为空。")

            # 1. 全量 Attention
            with torch.no_grad():
                try:
                    # === modify start: Handle LLaVA-OV CLS token ===
                    if model_type == "llava_ov":
                        # LLaVA-OV 每个片段的第一个 Token 是 CLS
                        # 我们使用 CLS 作为 Query，其余作为 Key (类似原始 LLaVA)
                        # q_slice: [H, N_total, D]
                        # k_slice: [H, N_total, D]
                        
                        # 检查切片长度
                        seq_len_slice = q_slice.shape[1]
                        if seq_len_slice > 1:
                            q_use = q_slice[:, 0:1, :] # [H, 1, D] (CLS)
                            k_use = k_slice[:, 1:, :]  # [H, N_patch, D] (Patches)
                        else:
                            # 只有 CLS 或异常，回退到自注意力平均
                            q_use = q_slice.mean(dim=1, keepdim=True)
                            k_use = k_slice

                        attn_logits = torch.matmul(q_use, k_use.transpose(-1, -2)) / math.sqrt(head_dim)
                        attn_weights = torch.softmax(attn_logits, dim=-1)
                        
                        # [H, 1, N_patch] -> mean over heads -> [1, N_patch] -> squeeze -> [N_patch]
                        attn_score_fine = attn_weights.mean(dim=0).squeeze(0)
                        
                        # Key: Patches only
                        attn_key_fine = k_use
                    # === modify end ===
                    else:
                        # Qwen Logic (Self-Attention Mean)
                        if q_slice.shape[-2] >= 6144:
                            q_use = q_slice.mean(dim=1, keepdim=True)
                        else:
                            q_use = q_slice
                        
                        attn_logits = torch.matmul(q_use, k_slice.transpose(-1, -2)) / math.sqrt(head_dim)
                        attn_weights = torch.softmax(attn_logits, dim=-1)
                        attn_score_fine = attn_weights.mean(dim=0).sum(dim=0)
                        attn_key_fine = k_slice

                    # 4. 空间合并 (Spatial Merging)
                    N_fine = attn_score_fine.shape[0]
                    merge_unit = spatial_merge_size**2
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    raise ValueError(f"Attention 计算失败: {e}, shape: {q_slice.shape}, {k_slice.shape}")

            if merge_unit > 1:
                # 边界处理: Padding
                remainder = N_fine % merge_unit
                if remainder != 0:
                    pad_len = merge_unit - remainder
                    attn_score_fine = F.pad(attn_score_fine, (0, pad_len), value=0.0)
                    # Pad Key on N dim (dim 1)
                    attn_key_fine = F.pad(attn_key_fine, (0, 0, 0, pad_len), value=0.0)

                # --- Score Merging (1D Sum) ---
                score_merged = attn_score_fine.view(-1, merge_unit).sum(dim=-1)

                # --- Key Merging (Mean) ---
                # [H, N_fine, D] -> [H, N_merged, merge_unit, D] -> mean(2)
                key_merged = attn_key_fine.view(
                    attn_key_fine.shape[0], -1, merge_unit, attn_key_fine.shape[-1]
                ).mean(dim=2)
            else:
                score_merged = attn_score_fine
                key_merged = attn_key_fine

            qwen_merged_scores.append(score_merged)
            qwen_merged_keys.append(key_merged)

        # 5. 全局重排序 (Global Reorder)
        if not qwen_merged_scores:
            return [], []

        total_scores = torch.cat(qwen_merged_scores, dim=0)  # [Total_Merged]
        total_keys = torch.cat(qwen_merged_keys, dim=1)  # [H, Total_Merged, D]

        if reverse_indices is None:
            # Qwen3 / LLaVA-OV 可能不需要显式 Reorder，或者 config 缺失
            pass
        else:
            if total_scores.shape[0] != reverse_indices.shape[0]:
                # 容错：如果长度不匹配（可能因 padding 导致），忽略 reorder
                pass
            else:
                total_scores = total_scores[reverse_indices]
                total_keys = total_keys[:, reverse_indices, :]

        # 6. Key: Mean over Heads
        total_keys = total_keys.mean(dim=0)  # [Total_Merged, D]

        # 7. 重新拆分回 List
        split_sizes = [t.shape[0] for t in qwen_merged_scores]
        if sum(split_sizes) > 0:
            scores_splits = torch.split(total_scores, split_sizes)
            keys_splits = torch.split(total_keys, split_sizes)
        else:
            scores_splits, keys_splits = [], []

        for s, k in zip(scores_splits, keys_splits):
            final_scores_list.append(s.unsqueeze(0))  # [1, N_merged]
            final_keys_list.append(k.unsqueeze(0))  # [1, N_merged, D]

    return final_scores_list, final_keys_list


def get_valid_content_mask(
    input_ids: torch.Tensor,
    start_seq: List[int] = None,
    end_token_id: int = 151645,
    forbidden_ids: Set[int] = None,
) -> torch.Tensor:
    """
    [v4.0 修改] 提取**最后一个视觉 Token 之后**到 End Token 之间的文本内容。
    """
    if forbidden_ids is None:
        forbidden_ids = {151654, 151655, 151656, 151652, 151653}

    original_shape = input_ids.shape
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    forbidden_tensor = torch.tensor(list(forbidden_ids), device=device)

    for b in range(batch_size):
        row_ids = input_ids[b]

        # 1. 找到所有视觉 Token
        is_visual = torch.isin(row_ids, forbidden_tensor)
        visual_indices = torch.where(is_visual)[0]

        # 2. 确定起点
        if len(visual_indices) > 0:
            start_idx = visual_indices[-1].item() + 1
        else:
            # 无图情况，保留全部 (从头开始)
            start_idx = 0

        # 3. 确定终点 (End Token)
        if start_idx < seq_len:
            remainder = row_ids[start_idx:]
            end_relative_indices = (remainder == end_token_id).nonzero(as_tuple=True)[0]

            if len(end_relative_indices) > 0:
                end_idx = start_idx + end_relative_indices[0].item()
            else:
                end_idx = seq_len

            if end_idx > start_idx:
                mask[b, start_idx:end_idx] = True

    # 4. 二次过滤
    is_forbidden = torch.isin(input_ids, forbidden_tensor)
    mask = mask & (~is_forbidden)

    return mask.view(original_shape)

def _regroup_tensors_by_count(
    source_list: List[torch.Tensor], 
    target_counts: List[int], 
    grid_list: Optional[torch.Tensor] = None
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """
    [v4.5 新增] 将物理分段的张量列表按逻辑分段长度进行重组聚合。
    用于对齐 ViT 物理输出 (cu_seqlens 维度) 与 LLM 逻辑输入 (input_ids 维度)。

    Args:
        source_list: 物理分段的张量列表，每个形状为 [1, N_i, ...]
        target_counts: 逻辑分段的长度列表 [M_1, M_2, ...]
        grid_list: 可选的物理分段元数据 [N_total_units, 3]

    Returns:
        Tuple: (重组后的张量列表, 重组后的元数据张量)
    """
    source_total = sum(t.shape[1] for t in source_list)
    target_total = sum(target_counts)

    if source_total != target_total:
        raise ValueError(
            f"[Regroup Error] 总数不匹配：物理 Token 总数({source_total}) "
            f"!= 逻辑 Token 总数({target_total})。"
        )

    regrouped_list = []
    regrouped_grids = []
    source_ptr = 0
    num_sources = len(source_list)

    for target_idx, target_len in enumerate(target_counts):
        current_group = []
        current_len = 0
        
        # 记录该组逻辑块对应的第一个物理单元的 Grid 信息
        if grid_list is not None and source_ptr < len(grid_list):
            regrouped_grids.append(grid_list[source_ptr])

        while current_len < target_len:
            if source_ptr >= num_sources:
                raise ValueError(f"[Regroup Error] 逻辑段 {target_idx} 需要更多 Token，但物理单元已耗尽。")
            
            src_tensor = source_list[source_ptr]
            src_len = src_tensor.shape[1]
            
            if current_len + src_len > target_len:
                raise ValueError(
                    f"[Regroup Error] 物理单元 {source_ptr} (长{src_len}) 导致逻辑段 {target_idx} 溢出 "
                    f"(当前已累积{current_len}, 目标{target_len})。无法进行物理切分。"
                )
            
            current_group.append(src_tensor)
            current_len += src_len
            source_ptr += 1
        
        # 聚合
        regrouped_list.append(torch.cat(current_group, dim=1))

    if source_ptr != num_sources:
        raise ValueError(f"[Regroup Error] 逻辑段处理完毕，但物理单元仍有剩余 ({num_sources - source_ptr} 个)。")

    final_grids = torch.stack(regrouped_grids) if regrouped_grids else None
    return regrouped_list, final_grids