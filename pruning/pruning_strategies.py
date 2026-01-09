# -*- coding: utf-8 -*-
"""
视觉Token剪枝策略库 (v1.2 - Refactored)

本文件作为剪枝策略的注册中心和入口点。实际的策略函数实现已被分散到
`pruning/strategies/` 子目录下的各个模块中。

v1.2 更新:
- [重构] 将所有策略函数实现迁移到 `pruning/strategies/` 子目录。
- [新增] 导入并注册了新的 `predictor_driven_pruning` 策略。
- [维护] 更新了 `_PRUNING_STRATEGIES` 注册表以引用导入的函数。

================================================================================
|                                **策略函数开发规范** |
================================================================================
所有剪枝策略函数必须遵循以下统一接口和规范：

1. 函数签名:
   def your_strategy_name(context: Dict[str, Any], **kwargs) -> torch.Tensor:

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
     例如，在配置 `{"method": "random", "params": {"ratio": 0.5}}` 中，
     kwargs 将会是 `{"ratio": 0.5}`。

3. 返回值 (Returns):
   - torch.Tensor: 一个布尔类型的张量 `keep_mask`。
     - 形状必须为 [batch_size, seq_len]，与输入的 `input_ids` 形状一致。
     - `True` 值表示保留该位置的Token，`False` 表示剪枝该位置的Token。
     - **【强制】** 策略函数必须保证所有非视觉（文本、特殊符号）Token对应的位置为 `True`。
"""
import time
import torch
from typing import Dict, Any, Callable

STRATEGY_TIMING_ENABLED = False
STRATEGY_TIME_LIST = []

# ================================================================================
# |                      **特殊Token定义 (全局常量)** |
# ================================================================================
# A. 可剪枝Token (Prunable Tokens)
VISION_PAD_ID = 151654  # 似乎未使用，但保留
IMAGE_PAD_ID = 151655
VIDEO_PAD_ID = 151656  # 视频支持可能需要

PRUNABLE_TOKEN_IDS = [IMAGE_PAD_ID, VIDEO_PAD_ID, VISION_PAD_ID]  # 更新为实际使用的

# B. 必须保留的特殊Token (Non-Prunable Special Tokens)
IM_START_ID = 151644
IM_END_ID = 151645
VISION_START_ID = 151652
VISION_END_ID = 151653
# 其他可能的特殊 token ID ...
# --------------------------------------------------------------------------------

# --- 从子模块导入策略函数 ---
from .strategies.basic import (
    override_pruning,
    random_pruning,
    baseline_pruning,
)
from .strategies.attention_based import special_token_based_attention_pruning
from .strategies.divprune import divprune
from .strategies.dart import dart_pruning
from .strategies.selector_strategy import (
    vision_selector_pruning,
)
from .strategies.idpruner import idpruner
from .strategies.vispruner import vispruner_pruning
from .strategies.scope import scope_pruning

# [修改] 从独立文件导入 HiPrune
from .strategies.hiprune import hiprune_pruning


from .strategies.utils import _extract_and_validate_vision_token_info

from .merging_strategies import MERGING_STRATEGIES


def func_wrapper(func, method_name: str) -> Callable:
    """
    打印debug信息使用
    """

    def wrapped_func(context: Dict[str, Any], **kwargs) -> torch.Tensor:
        prun_mask = func(context, **kwargs)
        vision_indices, non_vision_indices, _, _ = (
            _extract_and_validate_vision_token_info(context)
        )
        num_vision_tokens = len(vision_indices)
        pruned_vision_tokens = torch.sum(prun_mask == False).item()
        print("============================")
        print(
            f"Debug Info: {method_name} - 视觉Token总数: {num_vision_tokens}, 剪枝的视觉Token: {pruned_vision_tokens}"
        )
        print("============================")
        return prun_mask

    return wrapped_func


# --- 策略注册表 ---

_PRUNING_STRATEGIES: Dict[str, Callable] = {
    # Basic
    "baseline": baseline_pruning,
    "override": override_pruning,
    "random": random_pruning,
    "special_token_based_attention": special_token_based_attention_pruning,
    "divprune": divprune,
    "dart": dart_pruning,
    "hiprune": hiprune_pruning,
    "vision_selector": vision_selector_pruning,
    "vispruner": vispruner_pruning,
    "idpruner": idpruner,
    "scope": scope_pruning,
}

# debug printing
# _PRUNING_STRATEGIES = {k: func_wrapper(v, k) for k, v in _PRUNING_STRATEGIES.items()}

_PRUNING_STRATEGIES.update(MERGING_STRATEGIES)


def get_pruning_strategy(name: str) -> Callable:
    """
    根据名称从注册表中获取剪枝策略函数。

    Args:
        name (str): 策略的名称 (注册表中的键)。

    Returns:
        Callable: 对应的策略函数。

    Raises:
        ValueError: 如果指定的名称无效。
    """
    strategy = _PRUNING_STRATEGIES.get(name)
    if strategy is None:
        raise ValueError(
            f"Unknown pruning strategy: '{name}'. Available strategies are: {list(_PRUNING_STRATEGIES.keys())}"
        )
    
    # 如果计时开关打开，返回带计时的包装函数
    if STRATEGY_TIMING_ENABLED:
        def timed_strategy(context: Dict[str, Any], **kwargs) -> torch.Tensor:
            start_time = time.perf_counter()
            result = strategy(context, **kwargs)
            end_time = time.perf_counter()
            # 记录执行耗时 (转换为毫秒 ms)
            STRATEGY_TIME_LIST.append((end_time - start_time) * 1000)
            return result
        return timed_strategy

    return strategy
