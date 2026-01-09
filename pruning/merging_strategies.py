# -*- coding: utf-8 -*-
"""
================================================================================
|       Token 合并策略库 (pruning/merging_strategies.py) v2.3 (Strategy Registered) |
================================================================================
文件功能:
本文件定义了 Token 合并策略的框架和规范。与剪枝策略（选择性丢弃）不同，
合并策略旨在将“待移除”Token的信息融合到“保留”的Token中。

v2.3 更新:
- [新增] 导入并注册了 `vision_merger_merging` 策略，该策略由
  `pruning/strategies/vision_merger_strategy.py` 文件实现。
- [接口] 保持 v2.2 的函数签名和四元组返回类型不变。

================================================================================
|                         **合并策略函数开发规范** |
================================================================================
所有合并策略函数必须遵循以下统一接口和规范：

1. 函数签名:
   def your_merging_strategy_name(
       context: Dict[str, Any],
       **kwargs
   ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:

2. 参数 (Args):
   - context (Dict[str, Any]): 一个由模型内部包装器模块在推理时动态构建的字典。
     它包含了执行剪枝决策所需的所有上下文信息。以下是所有可能出现的键值对的详尽列表：

     [-- 基础信息 (始终提供) --]
       - "input_ids" (torch.Tensor): 模型的完整输入Token ID序列。
         形状: [batch_size, seq_len]。
       - "vision_token_mask" (torch.Tensor): 标识视觉Token位置的布尔掩码。
         形状: [seq_len]。
       - "image_grid_thw" (torch.Tensor): 输入图像的网格尺寸信息。
         形状: [num_images, 3]。
       - "video_grid_thw" (torch.Tensor): 输入视频的网格尺寸信息。
         形状: [num_videos, 3]。
       - "pos_emb_ids" (torch.Tensor): 用于计算旋转位置编码（RoPE）的完整位置ID。
         这是一个复杂的张量，包含了文本和视觉部分的不同位置信息。
         形状: [3, batch_size, seq_len]。
       - "text_pos_ids" (torch.Tensor or None): 仅包含文本部分的位置ID。
         形状: [batch_size, seq_len]。
       - "window_index" (torch.Tensor): ViT中用于窗口注意力的Token重排索引。
       - "cu_seqlens_full" (torch.Tensor): 用于分割多张图的累积序列长度，在重计算时使用。

     [-- 可选信息 (由剪枝配置中的 `need_*` 标志决定) --]
       - "feature_map" (torch.Tensor, 可选): 注意力计算之前的隐藏状态（hidden_states）。
         若配置中 `need_feature_map` 为 True 则提供。
         形状: [batch_size, seq_len, hidden_size]。
       - "pre_rope_q" / "pre_rope_k" (torch.Tensor, 可选): 应用RoPE之前的Q/K向量。
         若配置中 `need_pre_rope_qk` 为 True 则提供。
         形状: [batch_size, num_heads, seq_len, head_dim]。
       - "post_rope_q" / "post_rope_k" (torch.Tensor, 可选): 应用RoPE之后的Q/K向量。
         若配置中 `need_post_rope_qk` 为 True 则提供。
         形状: [batch_size, num_heads, seq_len, head_dim]。
       - "attn_map" (torch.Tensor, 可选): 注意力权重矩阵。
         若配置中 `need_attn_map` 为 True 则提供。
         形状: [batch_size, num_heads, seq_len, seq_len]。
       - "vit_attn_map_list_vit_{idx}" (List[torch.Tensor], 可选):
         一个列表，包含ViT在第 `idx` 层为每个图像生成的原始注意力图。
       - "vit_post_rope_q_vit_{idx}" / "vit_post_rope_k_vit_{idx}" (torch.Tensor, 可选):
         ViT在第 `idx` 层RoPE之后的Q/K向量，用于重计算注意力。
         若配置中 `need_vit_post_rope_qk` 为 True 则提供。
         形状: [seq_len, num_heads, head_dim]。

   - **kwargs (Dict): 一个字典，用于接收在主配置文件中为该策略定义的超参数。
     例如，在配置 `{"method": "my_merge", "params": {"ratio": 0.75}}` 中，
     kwargs 将会是 `{"ratio": 0.75}`。合并策略函数需要*自己*根据 `ratio`
     (或其他参数) 来决定每个视觉片段的目标数量 M_i。

3. 返回值 (Returns):
   - 一个包含四个元素的元组:
     (merge_weight_list, sizes_list_gpu, is_vision_list_gpu, fake_mask)

     1.  **merge_weight_list (List[torch.Tensor])**:
         - 一个 Python 列表，其中包含*每个*视觉片段的合并变换矩阵 T_i。
         - 如果 `is_vision_list_gpu` 中有 K 个 True，此列表应包含 K 个张量。
         - 每个张量 T_i 的形状为 `[M_i, N_i]` (float16)，其中:
           - N_i 是该视觉片段的原始 Token 数 (来自 `sizes_list_gpu`)。
           - M_i 是该视觉片段合并后的 Token 数 (由策略函数根据 kwargs 决定)。
         - 约束: 每个 T_i 的行和必须为 1 (torch.sum(T_i, dim=1) 必须全为 1)。
         - **注意**: 文本片段*不*在此列表中。

     2.  **sizes_list_gpu (torch.Tensor)**:
         - 一个 1D GPU 张量，包含 N 中所有片段 (文本和视觉) 的长度。
         - 此张量必须由策略函数*自己*根据 `context['input_ids']` 计算得出。
         - 形状: `[num_segments]`。

     3.  **is_vision_list_gpu (torch.Tensor)**:
         - 一个 1D GPU 布尔张量，标记 `sizes_list_gpu` 中的每个片段是否为视觉片段。
         - 此张量必须由策略函数*自己*根据 `context['input_ids']` 计算得出。
         - 形状: `[num_segments]`。

     4.  **fake_mask (torch.Tensor)**:
         - 一个 1D GPU 布尔张量，代表合并后的*整个序列*的“伪掩码”。
         - 形状: `[N_original]` (原始序列长度)。
         - **构建逻辑**:
           - 遍历 `sizes_list_gpu` 和 `is_vision_list_gpu`：
           - 如果 `is_vision` 为 False (文本片段)，掩码应包含 `L_j` 个 `True`。
           - 如果 `is_vision` 为 True (视觉片段)，掩码应包含 `M_i` 个 `True` 和
             `N_i - M_i` 个 `False`。
         - **目的**: 此掩码的 `sum()` 等于合并后的总序列长度，而 `(mask == False).sum()`
           等于总共被合并掉（等效于剪枝掉）的 Token 数量。这允许 PruningCache
           正确计算 `pruned_tokens` 并调整 KV 缓存和 `cache_position`。
           例如：`past_key_values.set_pruning_mask_for_layer("global", fake_mask)`
"""

import torch
from typing import Dict, Any, Callable, Tuple, List

from .strategies.visionzip import vision_zip_merging


# ================================================================================
# |                      **合并策略注册表 (Merger Registry)** |
# ================================================================================

# 这个字典将存储所有已实现的合并策略函数
# 键是策略的名称 (字符串)，值是对应的函数对象
#
# [v2.3] 注册了 vision_merger_merging 策略
MERGING_STRATEGIES: Dict[str, Callable] = {
    "vision_zip_merging": vision_zip_merging,  # [新增] 注册 VisionZip
}


# ================================================================================
# |                      **策略获取函数 (Strategy Getter)** |
# ================================================================================


def get_merging_strategy(
    name: str,
) -> Callable[
    [Dict[str, Any], Any],  # 输入参数为 context 和 **kwargs
    Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    根据名称从注册表中获取合并策略函数。

    Args:
        name (str): 策略的名称 (注册表中的键)。

    Returns:
        Callable: 对应的合并策略函数。
                  (该函数符合本文件顶部定义的 v2.2 规范)

    Raises:
        ValueError: 如果指定的名称无效。
    """
    strategy = MERGING_STRATEGIES.get(name)
    if strategy is None:
        if not MERGING_STRATEGIES:
            raise ValueError(
                f"合并策略注册表 (MERGING_STRATEGIES) 为空。无法找到策略: '{name}'。"
            )
        raise ValueError(
            f"未知的合并策略: '{name}'。可用的合并策略有: {list(MERGING_STRATEGIES.keys())}"
        )
    return strategy
