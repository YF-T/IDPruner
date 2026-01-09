# -*- coding: utf-8 -*-
"""
================================================================================
|             HiPrune 剪枝策略 (`pruning/strategies/hiprune.py`) v1.1          |
================================================================================
文件功能:
本文件实现了 HiPrune (Hierarchical Pruning) 策略，复刻自 Qwen2.5-VL 官方/社区实现。

核心逻辑:
1.  **双层依赖**: 结合浅层 (Object Layer) 和 深层 (Last Layer) 的注意力分数。
2.  **空间锚点 (Shallow)**: 在浅层选出锚点，并强制保留其 4-邻域 (上下左右)，形成十字形簇，
    以保留空间结构信息。
3.  **语义补全 (Deep)**: 在深层选出剩余的高分 Token，补全预算，同时屏蔽掉浅层已选的 Token。

参数 (kwargs):
    ratio (float): 剪枝率 (1 - Retention Ratio)。
    object_layer (int): 浅层层号 (用于提取空间结构)。
    last_vit_layer (int): 深层层号 (用于提取语义信息)。
    alpha (float): 浅层锚点预算占比 (0~1)。

v1.1 更新:
- [适配] 增加对 LLaVA 模型的支持，通过 sqrt(N) 推断宽度。
- [修复] Qwen 模型继续使用 grid_thw 计算宽度。
"""

import torch
import math
from typing import Dict, Any, List

from .utils import (
    _extract_and_validate_vision_token_info,
    _recompute_attention_maps_for_all_images,
    identify_model_architecture,  # [v1.1] 新增导入
)


def hiprune_pruning(context: Dict[str, Any], **kwargs) -> torch.Tensor:
    """
    HiPrune 策略函数。
    """
    # 1. 参数解析
    try:
        ratio = kwargs["ratio"]
        object_layer = kwargs["object_layer"]
        last_vit_layer = kwargs["last_vit_layer"]
        alpha = kwargs["alpha"]
    except KeyError as e:
        raise ValueError(f"HiPrune 策略缺少必要参数: {e}")

    # 2. 获取基础数据
    input_ids = context.get("input_ids")
    if input_ids is None:
        raise ValueError("Context 缺少 'input_ids'。")

    device = input_ids.device
    
    # 获取图像网格信息 (Qwen 需要)
    image_grid_thw = context.get("image_grid_thw") # [N_images, 3] (T, H, W)
    video_grid_thw = context.get("video_grid_thw")
    assert image_grid_thw is not None or video_grid_thw is not None, "HiPrune 需要 image_grid_thw 或 video_grid_thw。"
    assert image_grid_thw is None or video_grid_thw is None, "HiPrune 不支持同时使用 image_grid_thw 和 video_grid_thw。"
    grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
    

    spatial_merge_size = context.get("spatial_merge_size", 2)

    # [v1.1] 识别模型架构
    model_type = identify_model_architecture(context)

    # 3. 提取视觉 Token 分布
    vision_indices_global, non_vision_indices_global, _, num_tokens_per_image = (
        _extract_and_validate_vision_token_info(context)
    )

    if len(vision_indices_global) == 0:
        return torch.ones_like(input_ids, dtype=torch.bool)

    # 4. 获取双层注意力分数 (List[Tensor])
    q_shallow = context.get(f"vit_post_rope_q_vit_{object_layer}")
    k_shallow = context.get(f"vit_post_rope_k_vit_{object_layer}")
    q_deep = context.get(f"vit_post_rope_q_vit_{last_vit_layer}")
    k_deep = context.get(f"vit_post_rope_k_vit_{last_vit_layer}")

    if any(x is None for x in [q_shallow, k_shallow, q_deep, k_deep]):
        raise ValueError(
            f"HiPrune 需要浅层({object_layer})和深层({last_vit_layer})的 Q/K 张量。\n"
            "请检查配置中 'needs': {'need_vit_post_rope_qk': 'specific'} 是否正确应用。"
        )

    # 重计算分数
    shallow_scores_list, _ = _recompute_attention_maps_for_all_images(
        q_shallow, k_shallow, context
    )
    deep_scores_list, _ = _recompute_attention_maps_for_all_images(
        q_deep, k_deep, context
    )

    # --- [核心修改]: 使用重组逻辑替代原有的 min_len 截断 ---
    from .utils import _regroup_tensors_by_count
    
    # 对 shallow 分数和 grid 进行重组
    shallow_scores_list, regrouped_grid_thw = _regroup_tensors_by_count(
        shallow_scores_list, num_tokens_per_image, grid_thw
    )
    
    # 对 deep 分数进行重组 (忽略返回的 grid，因为是一致的)
    deep_scores_list, _ = _regroup_tensors_by_count(
        deep_scores_list, num_tokens_per_image, None
    )

    # 现在长度已经强制对齐
    num_images = len(num_tokens_per_image) 
    # -----------------------------------------------------

    try:
        # 切分全局索引
        vision_indices_split = torch.split(vision_indices_global, num_tokens_per_image)
    except Exception as e:
        print(f"Error splitting vision indices: {e}")
        return torch.ones_like(input_ids, dtype=torch.bool)

    all_kept_indices_global = []

    # 准备迭代器 (处理 grid_thw 可能为 None 的情况)
    iter_data = [shallow_scores_list, deep_scores_list, vision_indices_split]
    if grid_thw is not None:
        assert grid_thw.shape[0] == num_images, f"grid_thw 长度 {grid_thw.shape[0]} 不等于 num_images {num_images}"
        iter_data.append(grid_thw)
    else:
        raise ValueError("HiPrune 需要 grid_thw 信息。")

    print(f"shallow_scores_list={[scores.shape for scores in shallow_scores_list]}")
    print(f"deep_scores_list={[scores.shape for scores in deep_scores_list]}")
    print(f"vision_indices_split={[indices.shape for indices in vision_indices_split]}")

    # 5. 按图片循环处理 (核心逻辑)
    for img_idx, (shallow_score, deep_score, global_idx_map, grid) in enumerate(zip(*iter_data)):
        # shallow_score: [1, N] -> [N]
        shallow_score = shallow_score.squeeze(0)
        deep_score = deep_score.squeeze(0)
        N = shallow_score.shape[0]

        # 5.1 计算预算
        target_k = int(round(N * (1.0 - ratio)))
        if ratio < 1.0 and target_k == 0 and N > 0:
            target_k = 1
        
        if target_k >= N:
            all_kept_indices_global.append(global_idx_map)
            continue

        shallow_k = int(round((target_k * alpha) / 5.0))
        if alpha > 0 and shallow_k == 0 and target_k >= 5:
            shallow_k = 1
        
        shallow_indices_final = torch.tensor([], dtype=torch.long, device=device)

        # 5.2 第一阶段：浅层空间聚类
        if shallow_k > 0:
            # 选锚点
            _, anchor_indices = torch.topk(shallow_score, k=shallow_k)
            
            # [v1.1] 动态计算 Grid Width
            width = 0
            if "qwen" in model_type:
                if grid is None:
                    raise ValueError("HiPrune: Qwen 模型需要 grid_thw 信息。")
                width = int(grid[2].item() // spatial_merge_size)
            elif "llava" in model_type:
                # LLaVA 通常是正方形 Patch 网格，N = H * W, 假设 H=W
                width = int(math.sqrt(N))
                if width * width != N:
                    # 如果不是正方形，打印警告，但仍然尝试执行（可能会导致边缘邻居计算不准确）
                    print(f"Warning (HiPrune): LLaVA 图像 Token 数 ({N}) 不是完全平方数，Width 取 {width}。")
            else:
                raise ValueError(f"HiPrune 暂不支持模型架构: {model_type}")

            # 扩展邻居 (Cross Shape)
            # neighbors: idx, idx-1, idx+1, idx-width, idx+width
            neighbors = torch.cat([
                anchor_indices,
                anchor_indices - 1,
                anchor_indices + 1,
                anchor_indices - width,
                anchor_indices + width
            ])
            
            # 过滤非法索引
            neighbors = neighbors.clamp(0, N - 1)
            
            # 去重
            shallow_indices_final = torch.unique(neighbors)

        # 5.3 第二阶段：深层语义补全
        current_kept_count = shallow_indices_final.shape[0]
        deep_k = target_k - current_kept_count
        
        deep_indices_final = torch.tensor([], dtype=torch.long, device=device)

        if deep_k > 0:
            # 屏蔽已选
            deep_score_masked = deep_score.clone()
            if current_kept_count > 0:
                deep_score_masked[shallow_indices_final] = float('-inf')
            
            # 选 Top-K
            num_available = N - current_kept_count
            actual_deep_k = min(deep_k, num_available)
            
            if actual_deep_k > 0:
                _, deep_indices_final = torch.topk(deep_score_masked, k=actual_deep_k)

        # 5.4 合并
        final_local_indices = torch.cat([shallow_indices_final, deep_indices_final])
        
        # 映射回全局索引
        all_kept_indices_global.append(global_idx_map[final_local_indices])

    # 6. 构建全序列掩码
    if not all_kept_indices_global:
        return torch.ones_like(input_ids, dtype=torch.bool)

    kept_indices_tensor = torch.cat(all_kept_indices_global)
    final_indices = torch.cat([non_vision_indices_global.to(device), kept_indices_tensor])
    
    keep_mask = torch.zeros_like(input_ids.squeeze(0), dtype=torch.bool)
    keep_mask[final_indices] = True
    
    return keep_mask.unsqueeze(0)