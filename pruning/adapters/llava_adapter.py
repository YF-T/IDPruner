# -*- coding: utf-8 -*-
"""
================================================================================
|          LLaVA 模型剪枝适配器 (pruning/adapters/llava_adapter.py) v3.2            |
================================================================================
文件功能:
本文件实现了针对 LLaVA-1.5 (基于 Llama + CLIP) 的专属剪枝适配器。
代码结构严格对齐 HuggingFace v4.57.3 源代码，并修正了参数传递逻辑（移除不支持的 **kwargs）。

适配架构:
- Vision Tower: CLIP (modeling_clip.py) - 注意：CLIP源码中 forward 通常没有 kwargs
- LLM: Llama (modeling_llama.py)
- Top Level: Llava (modeling_llava.py)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable, Union, List, Tuple
import math
import traceback

# 导入 LLaVA, Llama, CLIP 的原始模块
from transformers.models.llava.modeling_llava import (
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaPreTrainedModel,
    LlavaMultiModalProjector,
    LlavaModelOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaDecoderLayer,
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    # ==== modify start: create_causal_mask is needed ====
    create_causal_mask,
    # ==== modify end ====
)
from transformers.models.clip.modeling_clip import (
    CLIPVisionModel,
    CLIPVisionTransformer,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPAttention,
    # ==== modify start: eager_attention_forward is needed for manual dispatch ====
    eager_attention_forward,
    # ==== modify end ====
)

# 导入辅助工具
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

# 导入项目内部模块
from pruning.adapters.base_adapter import BasePruningAdapter
from pruning.pruning_strategies import get_pruning_strategy
from pruning.pruning_cache import PruningCache
from pruning.pruning_modules import get_context_key

# 定义 LLaVA 的 Image Token ID (通常是 32000)
LLAVA_IMAGE_TOKEN_ID = 32000


# ================================================================================
# |                           1. CLIP (Vision) 包装类                            |
# ================================================================================


class Wrapped_CLIPAttention(CLIPAttention):
    """
    包装 CLIPAttention 以捕获中间状态。
    结构严格对齐 transformers/models/clip/modeling_clip.py 中的 CLIPAttention。
    注意：源码中 forward 没有 **kwargs。
    """

    def __init__(
        self, original_attn: CLIPAttention, pruning_conf: Dict[str, Any], layer_idx: int
    ):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_attn.__dict__)
        # ==== modify start: store pruning config ====
        self.pruning_config = pruning_conf
        self.layer_idx = layer_idx
        # ==== modify end ====

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        # ==== modify start: add context ====
        context: Dict[str, Any] = None,
        # ==== modify end ====
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, -1, self.head_dim).transpose(
            1, 2
        )
        keys = keys.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)

        # CLIP text model uses both `causal_attention_mask` and `attention_mask`
        # in case FA2 kernel is called, `is_causal` should be inferred from `causal_attention_mask`
        if self.config._attn_implementation == "flash_attention_2":
            self.is_causal = causal_attention_mask is not None
        else:
            if attention_mask is not None and causal_attention_mask is not None:
                attention_mask = attention_mask + causal_attention_mask
            elif causal_attention_mask is not None:
                attention_mask = causal_attention_mask

        # ==== modify start: capture pre/post rope (CLIP has no RoPE, so this is just Q/K) ====
        if context is not None:
            need_post_rope = self.pruning_config.get("needs", {}).get(
                "need_vit_post_rope_qk", False
            )
            q_key = get_context_key(
                "vit_post_rope_q", need_post_rope, "vit", self.layer_idx
            )
            k_key = get_context_key(
                "vit_post_rope_k", need_post_rope, "vit", self.layer_idx
            )

            if q_key:
                context[q_key] = queries
            if k_key:
                context[k_key] = keys
        # ==== modify end ====

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        # ==== modify start: force eager if attention map is needed ====
        need_attn = False
        if context is not None:
            need_attn = self.pruning_config.get("needs", {}).get(
                "need_vit_raw_attention", False
            )
            if need_attn:
                attention_interface = eager_attention_forward
        # ==== modify end ====

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,  # Argument from source
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            output_attentions=output_attentions,
        )

        # ==== modify start: capture attention map ====
        if context is not None and need_attn:
            attn_key = get_context_key(
                "vit_attn_map_list", need_attn, "vit", self.layer_idx
            )
            if attn_key and attn_weights is not None:
                context[attn_key] = attn_weights
        # ==== modify end ====

        attn_output = attn_output.reshape(
            batch_size, seq_length, embed_dim
        ).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class Prunable_CLIPEncoderLayer(CLIPEncoderLayer):
    """
    包装 CLIPEncoderLayer。
    结构严格对齐 transformers/models/clip/modeling_clip.py 中的 CLIPEncoderLayer。
    注意：源码中 forward 没有 **kwargs。
    """

    def __init__(
        self,
        original_layer: CLIPEncoderLayer,
        pruning_conf: Dict[str, Any],
        layer_idx: int,
    ):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_layer.__dict__)
        # ==== modify start: wrap self_attn ====
        self.layer_idx = layer_idx
        self.pruning_config = pruning_conf

        if not isinstance(self.self_attn, Wrapped_CLIPAttention):
            self.self_attn = Wrapped_CLIPAttention(
                self.self_attn, pruning_conf, layer_idx
            )
        # ==== modify end ====

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        # ==== modify start: add context ====
        context: Dict[str, Any] = None,
        # ==== modify end ====
    ) -> tuple[torch.FloatTensor]:

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # ==== modify start: pass context to self_attn ====
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            context=context,
        )
        # ==== modify end ====
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Prunable_CLIPEncoder(CLIPEncoder):
    """
    包装 CLIPEncoder 以支持剪枝上下文传递。
    注意：源码中 forward 没有 **kwargs。
    """

    def __init__(self, original_encoder: CLIPEncoder, pruning_config: Dict[str, Any]):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_encoder.__dict__)
        # ==== modify start: store config ====
        self.pruning_config = pruning_config
        # ==== modify end ====

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # ==== modify start: add context ====
        context: Dict[str, Any] = None,
        # ==== modify end ====
    ) -> BaseModelOutput:

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

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # ==== modify start: pass context to encoder_layer ====
            if isinstance(encoder_layer, Prunable_CLIPEncoderLayer):
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                    context=context,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )
            # ==== modify end ====

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class Prunable_CLIPVisionTransformer(CLIPVisionTransformer):
    """
    包装 CLIPVisionTransformer。
    注意：源码中 forward 没有 **kwargs。
    """

    def __init__(
        self, original_vision: CLIPVisionTransformer, pruning_config: Dict[str, Any]
    ):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_vision.__dict__)
        # ==== modify start: store config ====
        self.pruning_config = pruning_config
        # ==== modify end ====

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        # ==== modify start: add context ====
        context: Dict[str, Any] = None,
        # ==== modify end ====
    ) -> BaseModelOutputWithPooling:

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

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )
        hidden_states = self.pre_layrnorm(hidden_states)

        # ==== modify start: pass context to encoder ====
        # Ensure we call the wrapped encoder which accepts context
        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            context=context,
        )
        # ==== modify end ====

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 1:, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Prunable_CLIPVisionModel(CLIPVisionModel):
    """
    包装 CLIPVisionModel。
    注意：源码中 forward 没有 **kwargs。
    """

    def __init__(self, original_model: CLIPVisionModel, pruning_config: Dict[str, Any]):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_model.__dict__)
        self.pruning_config = pruning_config

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        # ==== modify start: match args and context ====
        context: Dict[str, Any] = None,
        # ==== modify end ====
    ) -> BaseModelOutputWithPooling:

        # ==== modify start: pass to vision_model (Prunable_CLIPVisionTransformer) ====
        # [新增] 根据配置捕获 pooler_output (Projection 前的特征)
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            context=context,
        )
        print("Prunable_CLIPVisionModel forward context:", context.keys())
        print("Prunable_CLIPVisionModel forward pruning_config:", self.pruning_config)
        if context is not None:
            needs = self.pruning_config.get("needs", {})
            if needs.get("need_vit_pooler_output", False):
                # CLIPVisionModel 的 forward 逻辑是:
                # outputs = self.vision_model(...)
                # pooled_output = outputs.pooler_output
                # return BaseModelOutputWithPooling(..., pooler_output=pooled_output)
                # self.vision_model (Transformer) 返回的 pooler_output 其实就是 [CLS] token 经过 post_layernorm 后的结果
                # 在 CLIPVisionModel 中，还会经过 self.visual_projection (如果是 CLIPModel)，
                # 但 LLaVA 的 vision_tower 是 CLIPVisionModel，通常没有 visual_projection 或者是分开的。
                # 无论如何，这里捕获的是 ViT 原始输出的 pooled 状态。
                print("Prunable_CLIPVisionModel forward context: vit_pooler_output")
                print(outputs.pooler_output.shape)
                context["vit_pooler_output"] = outputs.pooler_output

        return outputs
        # ==== modify end ====


# ================================================================================
# |                           2. Llama (LLM) 包装类                              |
# ================================================================================


class Wrapped_LlamaAttention(LlamaAttention):
    """
    包装 LlamaAttention。
    结构严格对齐 transformers/models/llama/modeling_llama.py 中的 LlamaAttention。
    注意：LlamaAttention 源码中有 **kwargs (Unpack[TransformersKwargs])，所以保留。
    """

    def __init__(
        self,
        original_attn: LlamaAttention,
        pruning_conf: Dict[str, Any],
        layer_idx: int,
    ):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_attn.__dict__)
        # ==== modify start: config ====
        self.pruning_config = pruning_conf
        self.layer_idx = layer_idx
        # ==== modify end ====

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # Llama signature
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # ==== modify start: custom args ====
        context: Dict[str, Any] = None,
        # ==== modify end ====
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # ==== modify start: custom forward injection (Recovery Wrapper) ====
        if context is not None and "custom_attention_forward_fns" in context:
            custom_fns = context["custom_attention_forward_fns"]
            if "Wrapped_LlamaAttention" in custom_fns:
                custom_fn = custom_fns["Wrapped_LlamaAttention"]
                # Explicitly pass self
                return custom_fn(
                    self,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    context=context,
                    **kwargs,
                )
        # ==== modify end ====

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # ==== modify start: capture pre-rope ====
        if context is not None:
            need_pre = self.pruning_config.get("needs", {}).get(
                "need_pre_rope_qk", False
            )
            if get_context_key("pre_rope_q", need_pre, "llm", self.layer_idx):
                context[
                    get_context_key("pre_rope_q", need_pre, "llm", self.layer_idx)
                ] = query_states
            if get_context_key("pre_rope_k", need_pre, "llm", self.layer_idx):
                context[
                    get_context_key("pre_rope_k", need_pre, "llm", self.layer_idx)
                ] = key_states
        # ==== modify end ====

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # ==== modify start: capture post-rope & feature map ====
        if context is not None:
            need_post = self.pruning_config.get("needs", {}).get(
                "need_post_rope_qk", False
            )
            if get_context_key("post_rope_q", need_post, "llm", self.layer_idx):
                context[
                    get_context_key("post_rope_q", need_post, "llm", self.layer_idx)
                ] = query_states
            if get_context_key("post_rope_k", need_post, "llm", self.layer_idx):
                context[
                    get_context_key("post_rope_k", need_post, "llm", self.layer_idx)
                ] = key_states

            need_feat = self.pruning_config.get("needs", {}).get(
                "need_feature_map", False
            )
            if get_context_key("feature_map", need_feat, "llm", self.layer_idx):
                context[
                    get_context_key("feature_map", need_feat, "llm", self.layer_idx)
                ] = hidden_states
        # ==== modify end ====

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        # ==== modify start: force eager for attn map capture ====
        if context is not None:
            need_map = self.pruning_config.get("needs", {}).get("need_attn_map", False)
            if need_map:
                attention_interface = eager_attention_forward
        # ==== modify end ====

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # ==== modify start: capture attn map ====
        if context is not None and attn_weights is not None:
            need_map = self.pruning_config.get("needs", {}).get("need_attn_map", False)
            if get_context_key("attn_map", need_map, "llm", self.layer_idx):
                context[
                    get_context_key("attn_map", need_map, "llm", self.layer_idx)
                ] = attn_weights
        # ==== modify end ====

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Prunable_LlamaDecoderLayer(LlamaDecoderLayer):
    """
    包装 LlamaDecoderLayer。
    LlamaDecoderLayer 源码中有 **kwargs (Unpack[TransformersKwargs])，保留。
    """

    def __init__(
        self,
        original_layer: LlamaDecoderLayer,
        pruning_conf: Dict[str, Any],
        layer_idx: int,
    ):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_layer.__dict__)
        # ==== modify start: wrap components ====
        self.layer_idx = layer_idx
        self.pruning_config = pruning_conf

        if not isinstance(self.self_attn, Wrapped_LlamaAttention):
            self.self_attn = Wrapped_LlamaAttention(
                self.self_attn, pruning_conf, layer_idx
            )

        self.pruning_fn = (
            get_pruning_strategy(pruning_conf["method"])
            if "method" in pruning_conf
            else None
        )
        # ==== modify end ====

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # ==== modify start: context ====
        context: Dict[str, Any] = None,
        # ==== modify end ====
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:

        # ==== modify start: custom layer forward injection ====
        if context is not None and "custom_layer_forward_fns" in context:
            custom_fns = context["custom_layer_forward_fns"]
            if "Prunable_LlamaDecoderLayer" in custom_fns:
                return custom_fns["Prunable_LlamaDecoderLayer"](
                    self,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    context=context,
                    **kwargs,
                )
        # ==== modify end ====

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # ==== modify start: pass context and args ====
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            context=context,
            **kwargs,
        )
        # ==== modify end ====

        hidden_states = residual + hidden_states

        # ==== modify start: Pruning Logic (Prefill Only) ====
        if (
            self.pruning_fn
            and isinstance(past_key_values, PruningCache)
            and past_key_values.is_prefill_stage(self.layer_idx, timing="after_update")
        ):
            shape_before = hidden_states.shape
            keep_mask = self.pruning_fn(
                context, **self.pruning_config.get("params", {})
            )

            if isinstance(keep_mask, torch.Tensor) and keep_mask.dtype == torch.bool:
                if keep_mask.dim() == 2:
                    keep_mask = keep_mask.squeeze(0)
                hidden_states = hidden_states[:, keep_mask, :]
                past_key_values.set_pruning_mask_for_layer(self.layer_idx, keep_mask)
                context["keep_mask"] = keep_mask

                # ==== modify start: enhanced debug output ====
                pruned_tokens = shape_before[1] - hidden_states.shape[1]
                print("=" * 50)
                print(f"[Layer Pruning] Layer {self.layer_idx}")
                print(f"Sequence length before: {shape_before[1]}")
                print(f"Sequence length after: {hidden_states.shape[1]}")
                print(f"Pruned tokens: {pruned_tokens}")
                print(
                    f"Keep ratio: {self.pruning_config.get('params', {}).get('ratio', None)}"
                )
                print(f"Hidden shape before: {shape_before}")
                print(f"Hidden shape after: {hidden_states.shape}")
                print(f"Pruning method: {self.pruning_config.get('method', 'unknown')}")
                print("=" * 50)
                # ==== modify end ====
        # ==== modify end ====

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Prunable_LlamaModel(LlamaModel):
    """
    包装 LlamaModel。
    LlamaModel 源码 forward 有 **kwargs。
    """

    def __init__(self, original_model: LlamaModel, pruning_config: Dict[str, Any]):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_model.__dict__)
        # ==== modify start: setup ====
        self.pruning_config = pruning_config
        self.pruning_fn = (
            get_pruning_strategy(pruning_config.get("method"))
            if "method" in pruning_config
            else None
        )
        print(f"[Prunable_LlamaModel] Pruning function: {self.pruning_fn}")
        # ==== modify end ====

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        # ==== modify start: context ====
        context: Dict[str, Any] = None,
        # ==== modify end ====
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:

        output_attentions = kwargs.get(
            "output_attentions", self.config.output_attentions
        )
        output_hidden_states = kwargs.get(
            "output_hidden_states", self.config.output_hidden_states
        )
        return_dict = kwargs.get("return_dict", self.config.use_return_dict)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            # ==== modify start: use PruningCache ====
            past_key_values = PruningCache(config=self.config)
            # ==== modify end ====

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # ==== modify start: Inject basic info into context ====
        if context is not None:
            context["inputs_embeds"] = inputs_embeds
            context["pos_emb_ids"] = position_ids.unsqueeze(0)  # Fake 3D for tools
            context["text_pos_ids"] = position_ids
            context["feature_map"] = inputs_embeds  # For Diversity etc.
        # ==== modify end ====

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # ==== modify start: GLOBAL PRUNING/MERGING LOGIC (Prefill) ====
        if (
            self.pruning_fn
            and isinstance(past_key_values, PruningCache)
            and past_key_values.is_prefill_stage(0, timing="before_update")
        ):
            hidden_states_shape_before = hidden_states.shape
            keep_mask = self.pruning_fn(
                context, **self.pruning_config.get("params", {})
            )

            # ==== modify start: debug output ====
            # Calculate image token info for debug output
            image_token_num_before = 0
            if "input_ids" in context:
                input_ids_tensor = context["input_ids"]
                image_token_num_before = torch.sum(
                    input_ids_tensor == LLAVA_IMAGE_TOKEN_ID
                ).item()

            # Simple Pruning
            if isinstance(keep_mask, torch.Tensor) and keep_mask.dtype == torch.bool:
                if keep_mask.dim() == 2:
                    keep_mask = keep_mask.squeeze(0)

                hidden_states = hidden_states[:, keep_mask, :]
                pruned_tokens = torch.sum(~keep_mask)

                # Debug output for pruning
                pruned_tokens_total = (
                    hidden_states_shape_before[1] - hidden_states.shape[1]
                )
                print("=" * 50)
                print(f"[Global Pruning] Layer 0 (Prefill)")
                print(f"Sequence length before: {hidden_states_shape_before[1]}")
                print(f"Sequence length after: {hidden_states.shape[1]}")
                print(f"Pruned tokens: {pruned_tokens_total}")
                print(f"Image tokens before pruning: {image_token_num_before}")
                print(
                    f"Keep ratio: {self.pruning_config.get('params', {}).get('ratio', None)}"
                )
                print(f"Hidden shape before: {hidden_states_shape_before}")
                print(f"Hidden shape after: {hidden_states.shape}")
                print("=" * 50)

                position_embeddings = (
                    position_embeddings[0][:, keep_mask, :],
                    position_embeddings[1][:, keep_mask, :],
                )
                if causal_mask is not None:
                    causal_mask = causal_mask[:, :, keep_mask, :][:, :, :, keep_mask]
                if cache_position is not None:
                    cache_position = torch.arange(
                        hidden_states.shape[1], device=hidden_states.device
                    )
                if position_ids is not None:
                    position_ids = position_ids[:, keep_mask]

                if "input_ids" in context:
                    context["input_ids"] = context["input_ids"][:, keep_mask]
                if context.get("pos_emb_ids") is not None:
                    context["pos_emb_ids"] = context["pos_emb_ids"][:, :, keep_mask]
                if context.get("text_pos_ids") is not None:
                    context["text_pos_ids"] = context["text_pos_ids"][
                        :, :-pruned_tokens
                    ]

                past_key_values.set_pruning_mask_for_layer("global", keep_mask)
            # ==== modify end ====

            # Merging
            elif isinstance(keep_mask, tuple) and len(keep_mask) == 4:
                (
                    merge_weight_list,
                    sizes_list_gpu,
                    is_vision_list_gpu,
                    merge_prun_mask,
                ) = keep_mask

                # ==== modify start: debug output for merging ====
                # Calculate image token info for debug output
                image_token_num_before = 0
                if "input_ids" in context:
                    input_ids_tensor = context["input_ids"]
                    image_token_num_before = torch.sum(
                        input_ids_tensor == LLAVA_IMAGE_TOKEN_ID
                    ).item()

                # Count vision segments for merging
                vision_segments_count = torch.sum(is_vision_list_gpu).item()
                total_segments = len(is_vision_list_gpu)

                update_position_ids = self.pruning_config.get("params", {}).get(
                    "update_position_ids", True
                )

                hidden_list = []
                position_ids_list = []
                sizes_list = sizes_list_gpu.tolist()
                embeds_split = torch.split(hidden_states, sizes_list, dim=1)
                pos_ids_split = torch.split(position_ids, sizes_list, dim=1)
                cur_img_id = 0

                for embed_part, pos_part, is_vision in zip(
                    embeds_split, pos_ids_split, is_vision_list_gpu
                ):
                    if is_vision:
                        merge_weight = (
                            merge_weight_list[cur_img_id]
                            .to(embed_part.device)
                            .to(embed_part.dtype)
                        )
                        if merge_weight.dim() == 2:
                            merge_weight = merge_weight.unsqueeze(0)
                        cur_img_id += 1
                        hidden_list.append(torch.matmul(merge_weight, embed_part))
                        # Merge Pos Ids
                        pos_part_float = pos_part.unsqueeze(
                            1
                        ).float()  # [bsz, 1, seq_len]
                        merged_pos = torch.matmul(
                            merge_weight.float(), pos_part_float.transpose(1, 2)
                        ).squeeze(2)
                        position_ids_list.append(merged_pos)
                    else:
                        hidden_list.append(embed_part)
                        position_ids_list.append(pos_part.float())

                hidden_states = torch.cat(hidden_list, dim=1)

                # Debug output for merging
                merged_tokens_total = (
                    hidden_states_shape_before[1] - hidden_states.shape[1]
                )
                print("=" * 50)
                print(f"[Global Merging] Layer 0 (Prefill)")
                print(f"Sequence length before: {hidden_states_shape_before[1]}")
                print(f"Sequence length after: {hidden_states.shape[1]}")
                print(f"Merged tokens: {merged_tokens_total}")
                print(f"Image tokens before merging: {image_token_num_before}")
                print(f"Vision segments: {vision_segments_count}/{total_segments}")
                print(
                    f"Merge ratio: {self.pruning_config.get('params', {}).get('ratio', None)}"
                )
                print(f"Hidden shape before: {hidden_states_shape_before}")
                print(f"Hidden shape after: {hidden_states.shape}")
                print(
                    f"Merging method: {self.pruning_config.get('text_model', {}).get('method', 'unknown')}"
                )
                print("=" * 50)
                # ==== modify end ====

                if update_position_ids:
                    position_ids = torch.cat(position_ids_list, dim=1).long()
                else:
                    position_ids = position_ids[:, merge_prun_mask]
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                pruned_tokens = torch.sum(~merge_prun_mask)
                if causal_mask is not None:
                    causal_mask = causal_mask[:, :, merge_prun_mask, :][
                        :, :, :, merge_prun_mask
                    ]
                if "input_ids" in context:
                    context["input_ids"] = context["input_ids"][:, merge_prun_mask]
                if context.get("pos_emb_ids") is not None:
                    context["pos_emb_ids"] = context["pos_emb_ids"][
                        :, :, merge_prun_mask
                    ]
                if context.get("text_pos_ids") is not None:
                    context["text_pos_ids"] = context["text_pos_ids"][
                        :, :-pruned_tokens
                    ]
                if cache_position is not None:
                    cache_position = torch.arange(
                        hidden_states.shape[1], device=hidden_states.device
                    )

                past_key_values.set_pruning_mask_for_layer("global", merge_prun_mask)

        # Decode Compensation
        if isinstance(
            past_key_values, PruningCache
        ) and not past_key_values.is_prefill_stage(0, timing="before_update"):
            if "global" in past_key_values.pruning_info:
                _, _, pruned_count_global = past_key_values.get_pruning_mask_for_layer(
                    "global"
                )
                if pruned_count_global > 0 and position_ids is not None:
                    position_ids = position_ids - pruned_count_global
        # ==== modify end ====

        # ==== modify start: save mask to context ====
        if context is not None:
            context["full_attention_mask"] = causal_mask
        # ==== modify end ====

        for layer_idx, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # ==== modify start: call decoder layer ====
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                # ==== modify start: pass context ====
                context=context,
                # ==== modify end ====
                **kwargs,
            )

            # Inter-layer sync
            if "keep_mask" in context:
                keep_mask = context.pop("keep_mask")
                if causal_mask is not None:
                    if causal_mask.dim() == 4:
                        causal_mask = causal_mask[:, :, keep_mask, :][
                            :, :, :, keep_mask
                        ]
                    else:
                        raise ValueError(
                            f"Invalid causal mask shape: {causal_mask.shape}"
                        )

                if position_ids is not None:
                    position_ids = position_ids[:, keep_mask]
                if cache_position is not None:
                    cache_position = cache_position[keep_mask]
                if position_embeddings is not None:
                    position_embeddings = (
                        position_embeddings[0][:, keep_mask, :],
                        position_embeddings[1][:, keep_mask, :],
                    )
                if "input_ids" in context:
                    context["input_ids"] = context["input_ids"][:, keep_mask]
                if "pos_emb_ids" in context:
                    context["pos_emb_ids"] = context["pos_emb_ids"][:, :, keep_mask]
                if "text_pos_ids" in context and context["text_pos_ids"] is not None:
                    context["text_pos_ids"] = context["text_pos_ids"][:, keep_mask]

            # Decode Sync for per-layer pruning
            if isinstance(
                past_key_values, PruningCache
            ) and not past_key_values.is_prefill_stage(
                layer_idx, timing="after_update"
            ):
                is_pruned, keep_mask, pruned_count = (
                    past_key_values.get_pruning_mask_for_layer(layer_idx)
                )
                if is_pruned and pruned_count > 0:
                    if position_ids is not None:
                        position_ids = position_ids - pruned_count
                    if cache_position is not None:
                        cache_position = cache_position - pruned_count
                    if (
                        causal_mask is not None
                        and causal_mask.shape[-1] > keep_mask.shape[-1]
                    ):
                        padding_shape = causal_mask.shape[-1] - keep_mask.shape[-1]
                        padding_true = torch.ones(
                            padding_shape,
                            dtype=keep_mask.dtype,
                            device=keep_mask.device,
                        )
                        effective_mask = torch.cat([keep_mask, padding_true], dim=-1)
                    else:
                        effective_mask = keep_mask
                    if causal_mask is not None:
                        causal_mask = causal_mask[:, :, :, effective_mask]
            # ==== modify end ====

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Prunable_LlavaModel_TopLevel(LlavaModel):
    """
    Prunable Top-Level LLaVA Model.
    LlavaModel.forward has **kwargs.
    """

    def __init__(self, original_model: LlavaModel, pruning_config: Dict[str, Any]):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_model.__dict__)
        # ==== modify start: config ====
        self.pruning_config = pruning_config
        # ==== modify end ====

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        # ==== modify start: add context ====
        context: Dict[str, Any] = None,
        # ==== modify end ====
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaModelOutputWithPast]:

        # ==== modify start: setup context ====
        if context is None:
            context = {}
        context["model_config"] = self.config
        context["input_ids"] = input_ids
        context["image_sizes"] = image_sizes

        # [新增] 注入 pixel_values 供辅助模型使用
        if pixel_values is not None:
            context["pixel_values"] = pixel_values
        # ==== modify end ====

        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            # ==== modify start: call get_image_features via self with context ====
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
                context=context,  # Pass context
            )
            # ==== modify end ====

            image_features = torch.cat(image_features, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

            # ==== modify start: capture vision mask ====
            context["vision_token_mask_for_all"] = special_image_mask
            # ==== modify end ====

        # ==== modify start: pass context to language model ====
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            context=context,
            **kwargs,
        )
        # ==== modify end ====

        return LlavaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    # ==== modify start: override get_image_features to pass context ====
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        image_sizes: Optional[torch.Tensor] = None,
        context: Dict[str, Any] = None,  # New arg
        **kwargs,
    ):
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
            )

        # kwargs = {k: v for k, v in kwargs.items() if v is not None and k != "context"} # Clean kwargs if passed to vision tower

        # Call vision tower with context.
        # VisionTower in LLaVA is CLIPVisionModel, which we wrapped.
        # It expects NO KWARGS in forward except defined ones?
        # Actually our Prunable_CLIPVisionModel.forward signature is:
        # (pixel_values, output_attentions, output_hidden_states, interpolate_pos_encoding, context)
        # We need to explicitly pass these.

        if isinstance(self.vision_tower, Prunable_CLIPVisionModel):
            image_outputs = self.vision_tower(
                pixel_values,
                output_hidden_states=True,
                context=context,
                # We don't pass arbitrary kwargs because CLIPVisionModel doesn't support them in source
                # But AutoModel.from_config creates it.
            )
        else:
            image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)

        # ... (Rest of logic same as source) ...
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [
                image_outputs.hidden_states[layer_idx]
                for layer_idx in vision_feature_layer
            ]
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature)

        if image_sizes is not None:
            split_sizes = [
                (height // self.vision_tower.patch_size)
                * (width // self.vision_tower.patch_size)
                for height, width in image_sizes
            ]
            image_features = torch.split(image_features.squeeze(0), split_sizes)
        else:
            image_features = list(image_features)
        return image_features

    # ==== modify end ====


# ================================================================================
# |                         3. LLaVA 适配器类                                    |
# ================================================================================


class LlavaPruningAdapter(BasePruningAdapter):
    """
    LLaVA-1.5 模型的剪枝适配器。
    """

    def wrap_model(self) -> torch.nn.Module:
        print("Activating LLaVA Pruning Plugin via Adapter...")
        model = self.model  # LlavaForConditionalGeneration
        config = self.config
        wrapped_components = set()

        # 防止 accelerate 干扰
        if hasattr(model, "_old_forward"):
            model._old_forward = None

        # 1. Wrap Top-Level Model (LlavaModel)
        if "vl_model" in config and not isinstance(
            model.model, Prunable_LlavaModel_TopLevel
        ):
            model.old_model["model"] = model.model
            model.model = Prunable_LlavaModel_TopLevel(
                model.model, config.get("vl_model", {})
            )
            print("  - Wrapped top-level LlavaModel.")
            wrapped_components.add("vl_model")

        # 2. Wrap Vision Tower (CLIP)
        if "vision_model" in config:
            vision_tower = model.model.vision_tower  # CLIPVisionModel

            # 2.1 Wrap Outer: CLIPVisionModel -> Prunable_CLIPVisionModel
            if not isinstance(vision_tower, Prunable_CLIPVisionModel):
                model.old_model["vision_tower_outer"] = vision_tower
                wrapped_vision_tower = Prunable_CLIPVisionModel(
                    vision_tower, config.get("vision_model", {})
                )
                model.model.vision_tower = wrapped_vision_tower
                print("  - Wrapped CLIPVisionModel (Outer).")
                wrapped_components.add("vision_model_outer")

            # 2.2 Wrap Middle: CLIPVisionTransformer -> Prunable_CLIPVisionTransformer
            vision_transformer = model.model.vision_tower.vision_model
            if not isinstance(vision_transformer, Prunable_CLIPVisionTransformer):
                model.old_model["vision_transformer_middle"] = vision_transformer
                # We need to wrap the transformer itself.
                wrapped_vision_transformer = Prunable_CLIPVisionTransformer(
                    vision_transformer, config.get("vision_model", {})
                )
                model.model.vision_tower.vision_model = wrapped_vision_transformer

                # # Also wrap the encoder inside the transformer to pass context
                # original_encoder = model.model.vision_tower.vision_model.encoder
                # model.model.vision_tower.vision_model.encoder = Prunable_CLIPEncoder(
                #     original_encoder, config.get("vision_model", {})
                # )

                print("  - Wrapped CLIPVisionTransformer (Middle) and CLIPEncoder.")
                wrapped_components.add("vision_model_middle")

            # 2.3 Wrap Inner: CLIPEncoder -> Prunable_CLIPEncoder
            vision_encoder = model.model.vision_tower.vision_model.encoder
            if not isinstance(vision_encoder, Prunable_CLIPEncoder):
                model.old_model["vision_encoder_inner"] = vision_encoder
                model.model.vision_tower.vision_model.encoder = Prunable_CLIPEncoder(
                    vision_encoder, config.get("vision_model", {})
                )
                print("  - Wrapped CLIPEncoder.")
                wrapped_components.add("vision_model_inner")

        # 3. Wrap Text Model (Llama) - LLM Backbone
        if "text_model" in config and not isinstance(
            model.model.language_model, Prunable_LlamaModel
        ):
            model.old_model["language_model"] = model.model.language_model
            model.model.language_model = Prunable_LlamaModel(
                model.model.language_model, config.get("text_model", {})
            )
            print("  - Wrapped LlamaModel (LLM Backbone).")
            wrapped_components.add("text_model")

        # 4. Wrap Vision Layers (CLIPEncoderLayer)
        if "vision_blocks" in config:
            if "vision_blocks" not in model.old_model:
                model.old_model["vision_blocks"] = {}

            # Access via wrapped chain
            encoder_layers = model.model.vision_tower.vision_model.encoder.layers
            for layer_idx_str, pruning_conf in config["vision_blocks"].items():
                layer_idx = int(layer_idx_str)
                original_layer = encoder_layers[layer_idx]
                if not isinstance(original_layer, Prunable_CLIPEncoderLayer):
                    model.old_model["vision_blocks"][layer_idx] = {
                        "layer": original_layer,
                        "self_attn": original_layer.self_attn,
                    }
                    encoder_layers[layer_idx] = Prunable_CLIPEncoderLayer(
                        original_layer, pruning_conf, layer_idx
                    )
                    print(f"  - Wrapped CLIP Encoder Layer {layer_idx}.")
                    wrapped_components.add(f"vision_block_{layer_idx}")

        # 5. Wrap Text Layers (LlamaDecoderLayer)
        if "decoder_layers" in config:
            if "decoder_layers" not in model.old_model:
                model.old_model["decoder_layers"] = {}

            decoder_layers = model.model.language_model.layers
            for layer_idx_str, pruning_conf in config["decoder_layers"].items():
                layer_idx = int(layer_idx_str)
                original_layer = decoder_layers[layer_idx]
                if not isinstance(original_layer, Prunable_LlamaDecoderLayer):
                    model.old_model["decoder_layers"][layer_idx] = {
                        "layer": original_layer,
                        "self_attn": original_layer.self_attn,
                    }
                    decoder_layers[layer_idx] = Prunable_LlamaDecoderLayer(
                        original_layer, pruning_conf, layer_idx
                    )
                    print(f"  - Wrapped Llama Decoder Layer {layer_idx}.")
                    wrapped_components.add(f"decoder_layer_{layer_idx}")

        print("Pruning plugin for LLaVA activated.")

        # --- 验证 ---
        try:
            assert hasattr(model, "_pruning_adapter"), "Adapter missing on model!"
            if "text_model" in wrapped_components:
                assert isinstance(model.model.language_model, Prunable_LlamaModel)
            if "vision_model_inner" in wrapped_components:
                assert isinstance(
                    model.model.vision_tower.vision_model,
                    Prunable_CLIPVisionTransformer,
                )
            if "vision_model_outer" in wrapped_components:
                assert isinstance(
                    model.model.vision_tower,
                    Prunable_CLIPVisionModel,
                )
            print("  - Wrapping verified.")
        except AssertionError as e:
            print(f"  - Wrapping verification failed: {e}")

        return model

    def unwrap_model(self) -> torch.nn.Module:
        print("Recovering original LLaVA model...")
        model = self.model
        if hasattr(model, "old_model") and model.old_model:
            old = model.old_model

            # 1. Restore Text Layers (Outer & Inner)
            if "decoder_layers" in old:
                layers = model.model.language_model.layers
                for idx, info in old["decoder_layers"].items():
                    layers[idx] = info["layer"]
                    layers[idx].self_attn = info["self_attn"]

            # 2. Restore Vision Layers (Outer & Inner)
            if "vision_blocks" in old:
                # Access current layers via potentially wrapped path
                layers = model.model.vision_tower.vision_model.encoder.layers
                for idx, info in old["vision_blocks"].items():
                    layers[idx] = info["layer"]
                    layers[idx].self_attn = info["self_attn"]

            # 3. Restore Models
            # Text Model
            if "language_model" in old:
                model.model.language_model = old["language_model"]

            # Vision Model (Reverse Order: Inner then Outer)
            if "vision_encoder_inner" in old:
                model.model.vision_tower.vision_model.encoder = old[
                    "vision_encoder_inner"
                ]
            if "vision_transformer_middle" in old:
                model.model.vision_tower.vision_model = old["vision_transformer_middle"]
            if "vision_tower_outer" in old:
                model.model.vision_tower = old["vision_tower_outer"]

            # Top-Level Model
            if "model" in old:
                model.model = old["model"]

            if hasattr(model, "_pruning_adapter"):
                del model._pruning_adapter
                print("  - Removed '_pruning_adapter' attribute.")

            # --- Strict Verification (Recursive) ---
            print("  - Performing strict recursive verification...")
            wrapper_classes = (
                Prunable_LlavaModel_TopLevel,
                Prunable_CLIPVisionModel,
                Prunable_CLIPVisionTransformer,
                Prunable_CLIPEncoder,
                Prunable_LlamaModel,
                Prunable_CLIPEncoderLayer,
                Prunable_LlamaDecoderLayer,
                Wrapped_CLIPAttention,
                Wrapped_LlamaAttention,
            )
            for name, module in model.named_modules():
                if isinstance(module, wrapper_classes):
                    raise AssertionError(
                        f"❌ Restoration FAILED: Module at path '{name}' is still an instance of wrapper class '{type(module).__name__}'!"
                    )

            assert not hasattr(
                model, "_pruning_adapter"
            ), "Model still has '_pruning_adapter' attribute!"

            model.old_model = {}
            print("✅ LLaVA model restored and verified.")
        else:
            print("  - No backup found.")
            assert not hasattr(
                model, "_pruning_adapter"
            ), "Model missing backup but still has '_pruning_adapter'!"

        return model
