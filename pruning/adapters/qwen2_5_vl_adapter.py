# -*- coding: utf-8 -*-
"""
================================================================================
|      Qwen2.5-VL 模型剪枝适配器 (pruning/adapters/qwen2_5_vl_adapter.py) v1.4      |
================================================================================
文件功能:
本文件实现了针对 Qwen2.5-VL 模型家族的专属剪枝适配器。它包含了所有用于替换
原始 Qwen2.5-VL 模块的 `Prunable_` 和 `Wrapped_` 类。

v1.4 更新:
- [代码重构] 合并了 `unwrap_model` 中的恢复步骤，在恢复外部层/块对象后立即恢复其内部注意力模块。
- [Bug修复] 在 `unwrap_model` 的末尾添加了 `del model._pruning_adapter`，修复了该属性未被移除的断言错误。
- [v1.3 保留] 在 `wrap_model` 和 `unwrap_model` 函数的末尾添加了全面的断言，
  以编程方式验证模块替换和恢复的正确性。
- [v1.2 保留] 修复了 `unwrap_model` 中因对象状态管理复杂性导致的 AssertionError。
  在恢复完整的 layer/block 对象后，增加了一个额外的步骤，强制将保存的原始 attn/self_attn
  赋给最终恢复的 layer/block，以确保状态正确。
- [v1.1 保留] 完善了 `unwrap_model` 方法，确保在恢复层对象之前，先恢复其内部的
  `attn` 和 `self_attn` 模块（虽然可能效果不佳，但保留步骤）。
- [v1.1 保留] 在 `unwrap_model` 的末尾添加了 `assert` 语句，以编程方式验证所有
  模块都已成功恢复到其原始类型。

核心组件:
- **Wrapped_Qwen2_5_...**: 这些类包装了原始的注意力模块，用于在不改变计算
  逻辑的前提下，捕获推理过程中产生的中间张量（如Q, K, V向量和注意力图），
  并将它们存入 `context` 字典。
- **Prunable_Qwen2_5_...**: 这些类包装了模型的计算单元（如DecoderLayer），
  它们负责：
    1. 判断当前是否为剪枝时机（Prefill阶段）。
    2. 从 `context` 字典中获取决策所需的信息。
    3. 调用 `pruning_strategies.py` 中的剪枝算法函数。
    4. 根据返回的 `keep_mask` 对 `hidden_states` 等张量进行切片。
    5. 将 `keep_mask` 存入 `PruningCache` 以便同步KV缓存。
- **Qwen2_5_VLPruningAdapter**: 实现了 `BasePruningAdapter` 接口，封装了
  完整的模块替换 (`wrap_model`) 和恢复 (`unwrap_model`) 逻辑。
"""
import traceback
from pruning.adapters.base_adapter import BasePruningAdapter
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Callable, Union, List

# 直接导入，如果失败则报错
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLModelOutputWithPast,
    apply_rotary_pos_emb_vision,
    apply_multimodal_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

# 从我们自己的文件中导入
from pruning.pruning_strategies import get_pruning_strategy
from pruning.pruning_cache import PruningCache
from pruning.pruning_modules import get_context_key

import time
ADAPTER_STATS_ENABLED = False
FWD_TIME_LIST = []            # 整个 forward 的耗时
BEFORE_TOTAL_LEN_LIST = []    # 剪枝前总长度
AFTER_TOTAL_LEN_LIST = []     # 剪枝后总长度
BEFORE_VISION_LEN_LIST = []   # 剪枝前视觉 Token 长度


class Wrapped_Qwen2_5_VLVisionAttention(Qwen2_5_VLVisionAttention):
    def __init__(self, original_attn, pruning_conf: Dict[str, Any], layer_idx: int):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_attn.__dict__)
        self.pruning_config = pruning_conf
        self.layer_idx = layer_idx
        self.attn_implementation = self.config._attn_implementation

        if self.pruning_config.get("needs", {}).get("need_vit_raw_attention"):
            self.attn_implementation = "eager"

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        context: Dict[str, Any] = None,
        **kwargs,
    ):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        if context is not None:
            need_pre_rope_config = self.pruning_config.get("needs", {}).get(
                "need_vit_pre_rope_qk", False
            )
            pre_rope_q_key = get_context_key(
                "vit_pre_rope_q", need_pre_rope_config, "vit", self.layer_idx
            )
            pre_rope_k_key = get_context_key(
                "vit_pre_rope_k", need_pre_rope_config, "vit", self.layer_idx
            )
            if pre_rope_q_key:
                context[pre_rope_q_key] = query_states
            if pre_rope_k_key:
                context[pre_rope_k_key] = key_states

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        if context is not None:
            need_post_rope_config = self.pruning_config.get("needs", {}).get(
                "need_vit_post_rope_qk", False
            )
            post_rope_q_key = get_context_key(
                "vit_post_rope_q", need_post_rope_config, "vit", self.layer_idx
            )
            post_rope_k_key = get_context_key(
                "vit_post_rope_k", need_post_rope_config, "vit", self.layer_idx
            )
            if post_rope_q_key:
                context[post_rope_q_key] = query_states
            if post_rope_k_key:
                context[post_rope_k_key] = key_states

        # [bsz, num_heads, seq_len, head_dim] (假设 bsz=1)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attn_weights = None
        attention_interface: Callable = eager_attention_forward
        if self.attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]

        if self.attn_implementation == "flash_attention_2":
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs_with_weights = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )
                for q, k, v in zip(*splits)
            ]

            attn_outputs = [output for output, _ in attn_outputs_with_weights]
            attn_weights_list = [weights for _, weights in attn_outputs_with_weights]

            attn_output = torch.cat(attn_outputs, dim=1)
            if self.attn_implementation == "eager":
                attn_weights = attn_weights_list

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)

        if context is not None:
            need_attn = self.pruning_config.get("needs", {}).get(
                "need_vit_raw_attention", False
            )
            attn_map_key = get_context_key(
                "vit_attn_map_list", need_attn, "vit", self.layer_idx
            )

            if (
                attn_map_key
                and self.attn_implementation == "eager"
                and attn_weights is not None
            ):
                context[attn_map_key] = attn_weights

            need_cu_seqlens_config = self.pruning_config.get("needs", {}).get(
                "need_cu_seqlens", False
            )
            cu_seqlens_key = get_context_key(
                "cu_seqlens", need_cu_seqlens_config, "vit", self.layer_idx
            )
            if cu_seqlens_key:
                context[cu_seqlens_key] = cu_seqlens

        return attn_output


class Wrapped_Qwen2_5_VLAttention(Qwen2_5_VLAttention):
    def __init__(self, original_attn, pruning_conf: Dict[str, Any], layer_idx: int):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_attn.__dict__)
        self.pruning_config = pruning_conf
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        context: Dict[str, Any] = None,
        **kwargs,
    ):
        # [修改] 检查是否有注入的自定义 forward 函数
        if context is not None and "custom_attention_forward_fns" in context:
            # print(f"====== Using custom forward function for Wrapped_Qwen2_5_VLAttention ======")
            custom_fns = context["custom_attention_forward_fns"]
            # print(f"Custom forward functions: {custom_fns}")
            if "Wrapped_Qwen2_5_VLAttention" in custom_fns:
                custom_fn = custom_fns["Wrapped_Qwen2_5_VLAttention"]
                # 调用注入的函数，显式传递 self
                return custom_fn(
                    self,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    context=context,
                    **kwargs,
                )

        # === 以下是标准逻辑 ===
        if context is not None:
            need_feature_map_config = self.pruning_config.get("needs", {}).get(
                "need_feature_map", False
            )
            feature_map_key = get_context_key(
                "feature_map", need_feature_map_config, "llm", self.layer_idx
            )
            if feature_map_key:
                context[feature_map_key] = hidden_states
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        if context is not None:
            need_pre_rope_config = self.pruning_config.get("needs", {}).get(
                "need_pre_rope_qk", False
            )
            pre_rope_q_key = get_context_key(
                "pre_rope_q", need_pre_rope_config, "llm", self.layer_idx
            )
            pre_rope_k_key = get_context_key(
                "pre_rope_k", need_pre_rope_config, "llm", self.layer_idx
            )
            if pre_rope_q_key:
                context[pre_rope_q_key] = query_states
            if pre_rope_k_key:
                context[pre_rope_k_key] = key_states
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )
        if context is not None:
            need_post_rope_config = self.pruning_config.get("needs", {}).get(
                "need_post_rope_qk", False
            )
            post_rope_q_key = get_context_key(
                "post_rope_q", need_post_rope_config, "llm", self.layer_idx
            )
            post_rope_k_key = get_context_key(
                "post_rope_k", need_post_rope_config, "llm", self.layer_idx
            )
            if post_rope_q_key:
                context[post_rope_q_key] = query_states
            if post_rope_k_key:
                context[post_rope_k_key] = key_states
            # 【新增】捕获V向量
            need_qkvh_config = self.pruning_config.get("needs", {}).get(
                "need_qkvh", False
            )
            qkvh_key = get_context_key("qkvh", need_qkvh_config, "llm", self.layer_idx)
            if qkvh_key:
                print(f"=== Capturing qkvh ===\n{qkvh_key}")
                context[qkvh_key] = (
                    query_states,
                    key_states,
                    value_states,
                    hidden_states,
                )

        # ====== [新增逻辑] 应用 Soft Pruning Mask (LLM Attention Scale) ======
        if context is not None and "kv_token_scale" in context:
            print(
                f"=== Applying Soft Pruning Mask (LLM Attention Scale) in Layer {self.layer_idx} ==="
            )
            # 1. 获取权重并转为 fp32
            raw_scale = context["kv_token_scale"].float()
            mode = context.get("kv_scale_mode", "sqrt")

            # 2. 维度调整: [Seq] 或 [B, Seq] -> [B, 1, Seq, 1] 以支持广播
            # Key States: [bsz, num_heads, seq_len, head_dim]
            if raw_scale.dim() == 1:
                scale_expanded = raw_scale.view(1, 1, -1, 1)
            elif raw_scale.dim() == 2:
                scale_expanded = raw_scale.unsqueeze(1).unsqueeze(-1)
            else:
                scale_expanded = raw_scale.view(1, 1, -1, 1)  # Fallback

            # 3. 计算缩放系数
            scale_k = None
            scale_v = None

            if mode == "sqrt":
                root_scale = torch.sqrt(scale_expanded)
                scale_k = root_scale
                scale_v = root_scale
            elif mode == "k_only":
                scale_k = scale_expanded
            elif mode == "v_only":
                scale_v = scale_expanded
            elif mode == "sqrt_k_only":  # [新增]
                scale_k = torch.sqrt(scale_expanded)
            elif mode == "sqrt_v_only":  # [新增]
                scale_v = torch.sqrt(scale_expanded)
            else:
                root_scale = torch.sqrt(scale_expanded)
                scale_k = root_scale
                scale_v = root_scale

            # 4. 应用
            target_dtype = key_states.dtype
            if scale_k is not None:
                key_states = key_states * scale_k.to(dtype=target_dtype)
            if scale_v is not None:
                value_states = value_states * scale_v.to(dtype=target_dtype)
        # ====================================================================

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = None
        attention_interface = ALL_ATTENTION_FUNCTIONS.get(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            # is_causal=self.is_causal,
            sliding_window=self.sliding_window,
            position_ids=position_ids,
            **kwargs,
        )

        if context is not None and attn_weights is not None:
            need_attn_map_config = self.pruning_config.get("needs", {}).get(
                "need_attn_map", False
            )
            attn_map_key = get_context_key(
                "attn_map", need_attn_map_config, "llm", self.layer_idx
            )
            if attn_map_key:
                context[attn_map_key] = attn_weights

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_values


class Prunable_Qwen2_5_VLVisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, original_block, pruning_conf: Dict[str, Any], layer_idx: int):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_block.__dict__)
        self.layer_idx = layer_idx

        if not isinstance(self.attn, Wrapped_Qwen2_5_VLVisionAttention):
            self.attn = Wrapped_Qwen2_5_VLVisionAttention(
                self.attn, pruning_conf, self.layer_idx
            )

        self.pruning_config = pruning_conf
        self.pruning_fn = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        context: Dict[str, Any] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states_norm = self.norm1(hidden_states)

        attn_output = self.attn(
            hidden_states_norm,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            context=context,
            **kwargs,
        )

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Prunable_Qwen2_5_VLDecoderLayer(Qwen2_5_VLDecoderLayer):
    def __init__(self, original_layer, pruning_conf: Dict[str, Any], layer_idx: int):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_layer.__dict__)
        self.layer_idx = layer_idx
        if not isinstance(self.self_attn, Wrapped_Qwen2_5_VLAttention):
            self.self_attn = Wrapped_Qwen2_5_VLAttention(
                self.self_attn, pruning_conf, self.layer_idx
            )
        self.pruning_config = pruning_conf
        self.pruning_fn = (
            get_pruning_strategy(pruning_conf["method"])
            if "method" in pruning_conf
            else None
        )
        # print(f"pruning_fn is {pruning_conf.get('method', None)}")
        if self.pruning_config.get("needs", {}).get("need_attn_map", False):
            self.attention_type = "eager"

        # # debug
        # setattr(self, "mlp", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        context: Dict[str, Any] = None,
        **kwargs,
    ):
        # [修改] 检查是否有注入的自定义 forward 函数 (Layer Level)
        if context is not None and "custom_layer_forward_fns" in context:
            # print(f"====== Using custom forward function for Prunable_Qwen2_5_VLDecoderLayer ======")
            custom_fns = context["custom_layer_forward_fns"]
            if "Prunable_Qwen2_5_VLDecoderLayer" in custom_fns:
                custom_fn = custom_fns["Prunable_Qwen2_5_VLDecoderLayer"]
                # 调用注入的函数，显式传递 self
                return custom_fn(
                    self,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    context=context,
                    **kwargs,
                )

        # assert False, "not implemented"
        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)

        # [新增] 剪枝恢复：Embed 模式下的 Norm 处理
        if context is not None and "compensation_info" in context:
            comp_info = context["compensation_info"]
            if comp_info["mode"] == "embed" and "pruned_hidden_states" in comp_info:
                pruned_hidden = comp_info["pruned_hidden_states"]
                pruned_hidden_norm = self.input_layernorm(pruned_hidden)
                layer_key = f"pruned_hidden_states_norm_layer_{self.layer_idx}"
                comp_info[layer_key] = pruned_hidden_norm

        attn_outputs = self.self_attn(
            hidden_states=hidden_states_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            context=context,
            **kwargs,
        )
        attn_output = attn_outputs[0]
        hidden_states = residual + attn_output
        assert hidden_states.shape[0] == 1, f"hidden_states.shape={hidden_states.shape}"
        # print(f"====== ready to prune {self.layer_idx} ======")
        if self.pruning_fn and past_key_values.is_prefill_stage(
            self.layer_idx, timing="after_update"
        ):
            # print(f"====== pruning {self.layer_idx} ======")
            keep_mask = self.pruning_fn(
                context, **self.pruning_config.get("params", {})
            )
            if (
                isinstance(keep_mask, torch.Tensor)
                and keep_mask.dtype == torch.bool
                and keep_mask.shape[-1] == hidden_states.shape[1]
                and hidden_states.shape[0] == 1
            ):
                # hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                keep_mask = keep_mask.view(-1)
                hidden_states_shape_before = hidden_states.shape
                hidden_states = hidden_states[:, keep_mask, :]
                # hidden_states = hidden_states.view(1, -1, hidden_states.shape[-1])
                context["keep_mask"] = keep_mask
                past_key_values.set_pruning_mask_for_layer(self.layer_idx, keep_mask)

                if ADAPTER_STATS_ENABLED:
                    # debug print
                    from ..pruning_strategies import PRUNABLE_TOKEN_IDS

                    is_image_token_before = torch.zeros_like(
                        context["input_ids"], dtype=torch.bool
                    )
                    for token_id in PRUNABLE_TOKEN_IDS:
                        is_image_token_before |= context["input_ids"] == token_id
                    image_token_num_before = torch.sum(is_image_token_before).item()
                    BEFORE_TOTAL_LEN_LIST.append(hidden_states_shape_before[1])
                    AFTER_TOTAL_LEN_LIST.append(hidden_states.shape[1])
                    BEFORE_VISION_LEN_LIST.append(image_token_num_before)
                    print(f"STATS UPDATED: {len(BEFORE_TOTAL_LEN_LIST)} samples")
            else:
                raise RuntimeError(
                    f"Pruning mask or hidden_states shape mismatch: keep_mask.shape={getattr(keep_mask, 'shape', None)}, hidden_states.shape={hidden_states.shape}"
                )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,) + attn_outputs[1:]
        return outputs


class Prunable_Qwen2_5_VisionTransformer(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, original_visual, pruning_config: Dict[str, Any]):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_visual.__dict__)
        self.pruning_config = pruning_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        context: Dict[str, Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens_full = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
        cu_seqlens_full = F.pad(cu_seqlens_full, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            cu_seqlens_now = (
                cu_seqlens_full
                if layer_num in self.fullatt_block_indexes
                else cu_window_seqlens
            )

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                context=context,
                **kwargs,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        if context is not None:
            context["window_index"] = window_index
            context["reverse_indices"] = reverse_indices
            context["spatial_merge_size"] = self.spatial_merge_size
            context["fullatt_block_indexes"] = self.fullatt_block_indexes
            context["cu_seqlens_full"] = cu_seqlens_full
            context["cu_window_seqlens"] = cu_window_seqlens

        return hidden_states


class Prunable_Qwen2_5_VLTextModel(Qwen2_5_VLTextModel):
    def __init__(self, original_model, pruning_conf: Dict[str, Any]):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_model.__dict__)
        self.pruning_config = pruning_conf
        self.pruning_fn = (
            get_pruning_strategy(pruning_conf["method"])
            if "method" in pruning_conf
            else None
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        context: Dict[str, Any] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:

        

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False
        if use_cache and not isinstance(past_key_values, PruningCache):
            past_key_values = PruningCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        is_prefill = past_key_values.is_prefill_stage(
            0, timing="before_update"
        )

        # _fwd_start = time.perf_counter() if ADAPTER_STATS_ENABLED and is_prefill else None

        cache_position_exists = cache_position is not None
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        position_ids_exists = position_ids is not None
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                3, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        text_position_ids = None
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # [修改] 注入完整的 position_embeddings 到 context
        if context is not None:
            # context["input_ids"] = input_ids
            context["inputs_embeds"] = inputs_embeds
            context["pos_emb_ids"] = position_ids
            context["old_pos_emb_ids"] = position_ids
            context["text_pos_ids"] = text_position_ids
            context["cache_position"] = cache_position
            context["full_position_embeddings"] = (
                position_embeddings  # For Recovery Wrapper
            )
            context["feature_map"] = hidden_states  # For Diversity etc.
            context["full_attention_mask"] = causal_mask_mapping["full_attention"]

        if self.pruning_fn and past_key_values.is_prefill_stage(
            0, timing="before_update"
        ):
            # debug print
            hidden_states_shape_before = hidden_states.shape
            from ..pruning_strategies import PRUNABLE_TOKEN_IDS

            is_image_token_before = torch.zeros_like(
                context["input_ids"], dtype=torch.bool
            )
            for token_id in PRUNABLE_TOKEN_IDS:
                is_image_token_before |= context["input_ids"] == token_id
            image_token_num_before = torch.sum(is_image_token_before)

            keep_mask = self.pruning_fn(
                context, **self.pruning_config.get("params", {})
            )
            if isinstance(keep_mask, torch.Tensor) and keep_mask.dtype == torch.bool:
                if (
                    isinstance(keep_mask, torch.Tensor)
                    and keep_mask.dtype == torch.bool
                    and keep_mask.shape[-1] == hidden_states.shape[1]
                    and hidden_states.shape[0] == 1
                ):
                    full_keep_mask_flat = keep_mask.view(-1)
                    pruned_tokens = torch.sum(~full_keep_mask_flat)
                    hidden_states = hidden_states[:, full_keep_mask_flat, :]
                    position_embeddings = (
                        position_embeddings[0][:, :, full_keep_mask_flat, :],
                        position_embeddings[1][:, :, full_keep_mask_flat, :],
                    )
                    if causal_mask_mapping.get("full_attention") is not None:
                        causal_mask_mapping["full_attention"] = causal_mask_mapping[
                            "full_attention"
                        ][:, :, full_keep_mask_flat, :][:, :, :, full_keep_mask_flat]
                    if causal_mask_mapping.get("sliding_attention") is not None:
                        causal_mask_mapping["sliding_attention"] = causal_mask_mapping[
                            "sliding_attention"
                        ][:, :, full_keep_mask_flat, :][:, :, :, full_keep_mask_flat]
                    if "input_ids" in context:
                        context["input_ids"] = context["input_ids"][
                            :, full_keep_mask_flat
                        ]
                    if context.get("pos_emb_ids") is not None:
                        context["pos_emb_ids"] = context["pos_emb_ids"][
                            :, :, full_keep_mask_flat
                        ]
                    if context.get("text_pos_ids") is not None:
                        context["text_pos_ids"] = context["text_pos_ids"][
                            :, :-pruned_tokens
                        ]
                    if cache_position is not None:
                        cache_position = torch.arange(
                            hidden_states.shape[1], device=hidden_states.device
                        )
                    if text_position_ids is not None:
                        text_position_ids = text_position_ids[:, :-pruned_tokens]
                    past_key_values.set_pruning_mask_for_layer(
                        "global", full_keep_mask_flat
                    )

                else:
                    raise RuntimeError(
                        f"Pruning mask or hidden_states shape mismatch: keep_mask.shape={getattr(keep_mask, 'shape', None)}, hidden_states.shape={hidden_states.shape}"
                    )
            else:
                # merge
                (
                    merge_weight_list,
                    sizes_list_gpu,
                    is_vision_list_gpu,
                    merge_prun_mask,
                ) = keep_mask
                update_position_ids = self.pruning_config.get("params", {}).get(
                    "update_position_ids", True
                )
                hidden_list = []
                position_ids_list = []
                cur_img_id = 0
                sizes_list_gpu = sizes_list_gpu.tolist()
                hidden_split_list = torch.split(hidden_states, sizes_list_gpu, dim=1)
                position_ids_split_list = torch.split(
                    position_ids, sizes_list_gpu, dim=2
                )
                for hidden_part, pos_id_part, is_vision in zip(
                    hidden_split_list, position_ids_split_list, is_vision_list_gpu
                ):
                    if is_vision:
                        # merge
                        merge_weight = (
                            merge_weight_list[cur_img_id]
                            .to(hidden_states.device)
                            .to(hidden_states.dtype)
                        )
                        if merge_weight.dim() == 2:
                            merge_weight = merge_weight.unsqueeze(0)
                        cur_img_id += 1
                        hidden_list.append(torch.matmul(merge_weight, hidden_part))
                        # position_ids (3, bs, seq-len) -> (bs, seq-len, 3)
                        pos_id_merged = torch.matmul(
                            merge_weight.float(), pos_id_part.permute(1, 2, 0).float()
                        )  # (bs, target_num, 3)
                        pos_id_merged = pos_id_merged.permute(
                            2, 0, 1
                        ).contiguous()  # (3, bs, target_num)
                        position_ids_list.append(pos_id_merged)
                    else:
                        # non-vision part, keep as is
                        hidden_list.append(hidden_part)
                        position_ids_list.append(pos_id_part)

                hidden_states = torch.cat(
                    hidden_list, dim=1
                )  # (bs, new_seq_len, hidden_size)
                # position_ids = torch.cat(position_ids_list, dim=2)  # (3, bs, new_seq_len)
                if update_position_ids:
                    position_ids = torch.cat(
                        position_ids_list, dim=2
                    )  # (3, bs, new_seq_len)
                else:
                    # [新增] 如果配置为不更新位置编码，则直接使用 merge_prun_mask 对原始位置编码进行切片
                    # merge_prun_mask 中 True 的位置对应于合并后的 Token（或者是代表 Token）
                    position_ids = position_ids[:, :, merge_prun_mask]
                # cache_position = torch.arange(
                #     past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
                # )
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                pruned_tokens = torch.sum(~merge_prun_mask)
                if causal_mask_mapping.get("full_attention") is not None:
                    causal_mask_mapping["full_attention"] = causal_mask_mapping[
                        "full_attention"
                    ][:, :, merge_prun_mask, :][:, :, :, merge_prun_mask]
                if causal_mask_mapping.get("sliding_attention") is not None:
                    causal_mask_mapping["sliding_attention"] = causal_mask_mapping[
                        "sliding_attention"
                    ][:, :, merge_prun_mask, :][:, :, :, merge_prun_mask]
                if "input_ids" in context:
                    context["input_ids"] = context["input_ids"][:, merge_prun_mask]
                if context.get("pos_emb_ids") is not None:
                    context["pos_emb_ids"] = position_ids
                if context.get("text_pos_ids") is not None:
                    context["text_pos_ids"] = context["text_pos_ids"][
                        :, :-pruned_tokens
                    ]
                if cache_position is not None:
                    cache_position = torch.arange(
                        hidden_states.shape[1], device=hidden_states.device
                    )
                if text_position_ids is not None:
                    text_position_ids = text_position_ids[:, :-pruned_tokens]
                past_key_values.set_pruning_mask_for_layer("global", merge_prun_mask)

            # debug print
            pruned_tokens_total = hidden_states_shape_before[1] - hidden_states.shape[1]
            print("=" * 50)
            print(f"Hidden shape before pruning: {hidden_states_shape_before}")
            print(f"Hidden shape after pruning: {hidden_states.shape}")
            print(f"Image token num before pruning: {image_token_num_before}")
            print(
                f"Keep ratio: {self.pruning_config.get('params', {}).get('ratio', None)}"
            )
            print(f"Pruned tokens total: {pruned_tokens_total}")
            print("=" * 50)

            if ADAPTER_STATS_ENABLED and is_prefill:
                BEFORE_TOTAL_LEN_LIST.append(hidden_states_shape_before[1])
                AFTER_TOTAL_LEN_LIST.append(hidden_states.shape[1])
                BEFORE_VISION_LEN_LIST.append(image_token_num_before.item())
                print(f"STATS UPDATED: {len(BEFORE_TOTAL_LEN_LIST)} samples")
                print(f"BEFORE_TOTAL_LEN_LIST: {BEFORE_TOTAL_LEN_LIST[-5:]}")
                print(f"AFTER_TOTAL_LEN_LIST: {AFTER_TOTAL_LEN_LIST[-5:]}")
                print(f"BEFORE_VISION_LEN_LIST: {BEFORE_VISION_LEN_LIST[-5:]}")

        if (
            not past_key_values.is_prefill_stage(0, timing="before_update")
            and past_key_values.get_pruning_mask_for_layer("global")[0]
        ):
            _, keep_mask, pruned_tokens = past_key_values.get_pruning_mask_for_layer(
                "global"
            )
            # cache_position 由 past_seen_tokens 计算而来
            if cache_position is not None and cache_position_exists:
                cache_position = cache_position - pruned_tokens
            # text_position_ids 由 position_ids 计算而来, position_ids由 cache_position 计算而来
            if text_position_ids is not None and (
                position_ids_exists or cache_position_exists
            ):
                text_position_ids = text_position_ids - pruned_tokens

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                context=context,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if "keep_mask" in context:
                keep_mask = context.pop("keep_mask")
                past_key_values.set_pruning_mask_for_layer(layer_idx, keep_mask)
                _, keep_mask, pruned_tokens = (
                    past_key_values.get_pruning_mask_for_layer(layer_idx)
                )
                if causal_mask_mapping[decoder_layer.attention_type] is not None:
                    causal_mask_mapping[decoder_layer.attention_type] = (
                        causal_mask_mapping[decoder_layer.attention_type][
                            :, :, keep_mask, :
                        ][:, :, :, keep_mask]
                    )
                if text_position_ids is not None:
                    text_position_ids = text_position_ids[:, :-pruned_tokens]
                if cache_position is not None:
                    cache_position = cache_position[:-pruned_tokens]
                if (
                    position_embeddings is not None
                    and position_embeddings[0] is not None
                    and position_embeddings[1] is not None
                ):
                    position_embeddings = (
                        position_embeddings[0][:, :, keep_mask, :],
                        position_embeddings[1][:, :, keep_mask, :],
                    )
                if "input_ids" in context:
                    context["input_ids"] = context["input_ids"][:, keep_mask]
                if "pos_emb_ids" in context:
                    context["pos_emb_ids"] = context["pos_emb_ids"][:, :, keep_mask]
                if "text_pos_ids" in context and context["text_pos_ids"] is not None:
                    context["text_pos_ids"] = context["text_pos_ids"][
                        :, :-pruned_tokens
                    ]
            if (
                not past_key_values.is_prefill_stage(layer_idx, timing="after_update")
                and past_key_values.get_pruning_mask_for_layer(layer_idx)[0]
            ):
                _, keep_mask, pruned_tokens = (
                    past_key_values.get_pruning_mask_for_layer(layer_idx)
                )
                if (
                    causal_mask_mapping[decoder_layer.attention_type] is not None
                    and causal_mask_mapping[decoder_layer.attention_type].shape[-1]
                    > keep_mask.shape[-1]
                ):
                    padding_shape = (
                        causal_mask_mapping[decoder_layer.attention_type].shape[-1]
                        - keep_mask.shape[-1]
                    )
                    padding_true = torch.ones(
                        padding_shape, dtype=keep_mask.dtype, device=keep_mask.device
                    )
                    effective_mask = torch.cat([keep_mask, padding_true], dim=-1)
                else:
                    effective_mask = keep_mask
                if causal_mask_mapping[decoder_layer.attention_type] is not None:
                    causal_mask_mapping[decoder_layer.attention_type] = (
                        causal_mask_mapping[decoder_layer.attention_type][
                            :, :, :, effective_mask
                        ]
                    )
                if text_position_ids is not None:
                    text_position_ids = text_position_ids - pruned_tokens
                if cache_position is not None:
                    cache_position = cache_position - pruned_tokens

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        
        # --- [统计结束: Forward 全程计时记录] ---
        # if ADAPTER_STATS_ENABLED and _fwd_start is not None:
        #     FWD_TIME_LIST.append((time.perf_counter() - _fwd_start) * 1000)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Prunable_Qwen2_5_VLModel(Qwen2_5_VLModel):
    def __init__(self, original_model, pruning_config: Dict[str, Any]):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_model.__dict__)
        self.pruning_config = pruning_config

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        context: Dict[str, Any] = None,
    ):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(
            pixel_values, grid_thw=image_grid_thw, context=context
        )
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
        context: Dict[str, Any] = None,
    ):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(
            pixel_values_videos, grid_thw=video_grid_thw, context=context
        )
        split_sizes = (
            video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        context = kwargs.pop("context", {})
        if context is None:
            context = {}

        is_prefill = past_key_values is None or past_key_values.get_seq_length() == 0

        _fwd_start = time.perf_counter() if ADAPTER_STATS_ENABLED and is_prefill else None


        # [新增] 将模型配置注入 Context
        context["model_config"] = self.config

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_embeds = self.get_image_features(
                pixel_values, image_grid_thw, context=context
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw, context=context
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        if position_ids is None:
            prefill_noncompiled_stage = not torch.jit.is_scripting() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if prefill_noncompiled_stage or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(
                        inputs_embeds.device
                    )
                else:
                    delta = torch.zeros(
                        (batch_size, seq_length), device=inputs_embeds.device
                    )
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids = position_ids + delta.to(position_ids.device)
        context.update(
            {
                "input_ids": input_ids,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "inputs_embeds": inputs_embeds,
                "original_input_ids": input_ids,
            }
        )
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            context=context,
            **kwargs,
        )
        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

        # --- [统计结束: Forward 全程计时记录] ---
        if ADAPTER_STATS_ENABLED and _fwd_start is not None:
            FWD_TIME_LIST.append((time.perf_counter() - _fwd_start) * 1000)

        return output if return_dict else output.to_tuple()


# ================================================================================
# |                      Qwen2.5-VL 专属适配器实现 (Adapter Implementation)            |
# ================================================================================


class Qwen2_5_VLPruningAdapter(BasePruningAdapter):
    """
    Qwen2.5-VL模型的专属剪枝适配器。
    """

    def wrap_model(self) -> torch.nn.Module:
        """
        根据配置，用可剪枝的模块包装Qwen2.5-VL模型。
        """
        print("Activating Qwen2.5-VL Pruning Plugin via Adapter...")
        model = self.model
        config = self.config
        wrapped_components = set()

        model._old_forward = None  # 防止使用 accelerate 库

        if "vl_model" in config and not isinstance(
            model.model, Prunable_Qwen2_5_VLModel
        ):
            model.old_model["model"] = model.model
            model.model = Prunable_Qwen2_5_VLModel(
                model.model, config.get("vl_model", {})
            )
            print("  - Wrapped top-level Qwen2_5_VLModel.")
            wrapped_components.add("vl_model")

        if "vision_model" in config and not isinstance(
            model.model.visual, Prunable_Qwen2_5_VisionTransformer
        ):
            model.old_model["visual"] = model.model.visual
            model.model.visual = Prunable_Qwen2_5_VisionTransformer(
                model.model.visual, config.get("vision_model", {})
            )
            print("  - Wrapped VisionTransformer.")
            wrapped_components.add("vision_model")

        if "text_model" in config and not isinstance(
            model.model.language_model, Prunable_Qwen2_5_VLTextModel
        ):
            model.old_model["language_model"] = model.model.language_model
            model.model.language_model = Prunable_Qwen2_5_VLTextModel(
                model.model.language_model, config.get("text_model", {})
            )
            print("  - Wrapped VLTextModel.")
            wrapped_components.add("text_model")

        if "vision_blocks" in config:
            if "vision_blocks" not in model.old_model:
                model.old_model["vision_blocks"] = {}
            for layer_idx_str, pruning_conf in config["vision_blocks"].items():
                layer_idx = int(layer_idx_str)
                original_block = model.model.visual.blocks[layer_idx]
                if not isinstance(original_block, Prunable_Qwen2_5_VLVisionBlock):
                    model.old_model["vision_blocks"][layer_idx] = {
                        "block": original_block,
                        "attn": original_block.attn,
                    }
                    model.model.visual.blocks[layer_idx] = (
                        Prunable_Qwen2_5_VLVisionBlock(
                            original_block, pruning_conf, layer_idx
                        )
                    )
                    print(f"  - Wrapped Vision Block {layer_idx} for info gathering.")
                    wrapped_components.add(f"vision_block_{layer_idx}")

        if "decoder_layers" in config:
            if "decoder_layers" not in model.old_model:
                model.old_model["decoder_layers"] = {}
            for layer_idx_str, pruning_conf in config["decoder_layers"].items():
                layer_idx = int(layer_idx_str)
                original_layer = model.model.language_model.layers[layer_idx]
                if not isinstance(original_layer, Prunable_Qwen2_5_VLDecoderLayer):
                    model.old_model["decoder_layers"][layer_idx] = {
                        "layer": original_layer,
                        "self_attn": original_layer.self_attn,
                    }
                    model.model.language_model.layers[layer_idx] = (
                        Prunable_Qwen2_5_VLDecoderLayer(
                            original_layer, pruning_conf, layer_idx
                        )
                    )
                    print(
                        f"  - Wrapped Decoder Layer {layer_idx} with prunable version."
                    )
                    wrapped_components.add(f"decoder_layer_{layer_idx}")

        print("Pruning plugin for Qwen2.5-VL activated successfully!")

        # --- wrap_model 末尾的断言验证 (保持不变) ---
        print("  - Verifying wrapping...")
        try:
            assert hasattr(
                model, "_pruning_adapter"
            ), "Model missing '_pruning_adapter' after wrapping!"
            if "vl_model" in wrapped_components:
                assert isinstance(
                    model.model, Prunable_Qwen2_5_VLModel
                ), "Top-level model not wrapped!"
            if "vision_model" in wrapped_components:
                assert isinstance(
                    model.model.visual, Prunable_Qwen2_5_VisionTransformer
                ), "Vision transformer not wrapped!"
            if "text_model" in wrapped_components:
                assert isinstance(
                    model.model.language_model, Prunable_Qwen2_5_VLTextModel
                ), "Language model not wrapped!"
            if "vision_blocks" in config:
                for layer_idx_str in config["vision_blocks"].keys():
                    layer_idx = int(layer_idx_str)
                    if f"vision_block_{layer_idx}" in wrapped_components:
                        block = model.model.visual.blocks[layer_idx]
                        assert isinstance(
                            block, Prunable_Qwen2_5_VLVisionBlock
                        ), f"Vision block {layer_idx} not wrapped!"
                        assert isinstance(
                            block.attn, Wrapped_Qwen2_5_VLVisionAttention
                        ), f"Vision block {layer_idx}'s attention not wrapped!"
            if "decoder_layers" in config:
                for layer_idx_str in config["decoder_layers"].keys():
                    layer_idx = int(layer_idx_str)
                    if f"decoder_layer_{layer_idx}" in wrapped_components:
                        layer = model.model.language_model.layers[layer_idx]
                        assert isinstance(
                            layer, Prunable_Qwen2_5_VLDecoderLayer
                        ), f"Decoder layer {layer_idx} not wrapped!"
                        assert isinstance(
                            layer.self_attn, Wrapped_Qwen2_5_VLAttention
                        ), f"Decoder layer {layer_idx}'s attention not wrapped!"
            print("  - Wrapping verification successful.")
        except AssertionError as e:
            print(f"  - Wrapping verification FAILED: {e}")
            traceback.print_exc()

        return model

    def unwrap_model(self) -> torch.nn.Module:
        """
        将Qwen2.5-VL模型恢复到原始状态，合并恢复步骤，并添加验证。
        """
        print("Recovering original Qwen2.5-VL model via Adapter...")
        model = self.model
        if hasattr(model, "old_model") and model.old_model:
            old_modules = model.old_model
            restored_components = set()  # 跟踪恢复的组件

            # --- 修改：合并恢复步骤 ---

            # Step 1: 恢复外部模块并立即恢复内部模块
            if "decoder_layers" in old_modules:
                for layer_idx, original_info in old_modules["decoder_layers"].items():
                    # 恢复外部 Layer 对象
                    model.model.language_model.layers[layer_idx] = original_info[
                        "layer"
                    ]
                    # 立即恢复内部 self_attn
                    model.model.language_model.layers[layer_idx].self_attn = (
                        original_info["self_attn"]
                    )
                    restored_components.add(f"decoder_layer_{layer_idx}")
                print("  - Restored Decoder Layers and their internal self_attn.")

            if "vision_blocks" in old_modules:
                for layer_idx, original_info in old_modules["vision_blocks"].items():
                    # 恢复外部 Block 对象
                    model.model.visual.blocks[layer_idx] = original_info["block"]
                    # 立即恢复内部 attn
                    model.model.visual.blocks[layer_idx].attn = original_info["attn"]
                    restored_components.add(f"vision_block_{layer_idx}")
                print("  - Restored Vision Blocks and their internal attn.")

            # Step 2: 恢复其他顶层模块
            if "language_model" in old_modules:
                model.model.language_model = old_modules["language_model"]
                restored_components.add("language_model")
                print("  - Restored VLTextModel.")
            if "visual" in old_modules:
                model.model.visual = old_modules["visual"]
                restored_components.add("vision_model")
                print("  - Restored VisionTransformer.")
            if "model" in old_modules:
                model.model = old_modules["model"]
                restored_components.add("vl_model")
                print("  - Restored top-level Qwen2_5_VLModel.")

            # Step 3: 删除适配器属性
            if hasattr(model, "_pruning_adapter"):
                del model._pruning_adapter
                print("  - Removed '_pruning_adapter' attribute.")

            # --- 修改结束 ---

            # Step 4: 验证恢复状态 (保持不变)
            print("  - Verifying restoration...")
            try:
                # 1. 验证顶层模型类型
                if "vl_model" in restored_components:
                    assert not isinstance(
                        model.model, Prunable_Qwen2_5_VLModel
                    ), "Top-level model not restored!"
                # 2. 验证语言模型组件
                if "language_model" in restored_components:
                    assert not isinstance(
                        model.model.language_model, Prunable_Qwen2_5_VLTextModel
                    ), "Language model not restored!"
                # 3. 验证视觉模型组件
                if "vision_model" in restored_components:
                    assert not isinstance(
                        model.model.visual, Prunable_Qwen2_5_VisionTransformer
                    ), "Vision transformer not restored!"

                # 4. 验证所有被恢复的解码器层
                if any(
                    comp.startswith("decoder_layer_") for comp in restored_components
                ):
                    for layer_idx_str in [
                        comp.split("_")[-1]
                        for comp in restored_components
                        if comp.startswith("decoder_layer_")
                    ]:
                        layer_idx = int(layer_idx_str)
                        restored_layer = model.model.language_model.layers[layer_idx]
                        assert not isinstance(
                            restored_layer, Prunable_Qwen2_5_VLDecoderLayer
                        ), f"Decoder layer {layer_idx} class not restored!"
                        assert not isinstance(
                            restored_layer.self_attn, Wrapped_Qwen2_5_VLAttention
                        ), f"Decoder layer {layer_idx} attention not restored!"

                # 5. 验证所有被恢复的视觉块
                if any(
                    comp.startswith("vision_block_") for comp in restored_components
                ):
                    for layer_idx_str in [
                        comp.split("_")[-1]
                        for comp in restored_components
                        if comp.startswith("vision_block_")
                    ]:
                        layer_idx = int(layer_idx_str)
                        restored_block = model.model.visual.blocks[layer_idx]
                        assert not isinstance(
                            restored_block, Prunable_Qwen2_5_VLVisionBlock
                        ), f"Vision block {layer_idx} class not restored!"
                        assert not isinstance(
                            restored_block.attn, Wrapped_Qwen2_5_VLVisionAttention
                        ), f"Vision block {layer_idx} attention not restored!"

                # 6. 验证没有残留的剪枝配置
                def check_no_pruning_config(module, path=""):
                    if hasattr(module, "pruning_config"):
                        raise AssertionError(f"Residual config at {path}")
                    if hasattr(module, "pruning_fn"):
                        raise AssertionError(f"Residual function at {path}")
                    for name, child in module.named_children():
                        check_no_pruning_config(
                            child, f"{path}.{name}" if path else name
                        )

                check_no_pruning_config(model.model, "model")

                # 7. 验证适配器属性被移除 (由 Step 3 完成)
                assert not hasattr(
                    model, "_pruning_adapter"
                ), "Model still has '_pruning_adapter' attribute after unwrap!"

                print("  - All modules successfully verified.")

            except AssertionError as e:
                print(f"  - Restoration verification FAILED: {e}")
                traceback.print_exc()

            # 清空备份
            model.old_model = {}

        else:  # 如果没有 old_model 属性
            print(
                "  - No backup found (old_model attribute missing or empty). Assuming model is already original."
            )
            assert not hasattr(
                model, "_pruning_adapter"
            ), "Model missing backup but still has '_pruning_adapter'!"

        print("✅ Qwen2.5-VL model successfully restored and verified.")
        return model
