# -*- coding: utf-8 -*-
"""
================================================================================
|     LLaVA-OneVision-1.5 剪枝适配器 (pruning/adapters/llava_ov_1_5_adapter.py)    |
================================================================================
文件功能:
本文件定义了针对 LLaVA OneVision 1.5 模型的专属适配器。
代码严格对齐 LLaVA OneVision 1.5 的源代码结构，通过继承原始类实现非侵入式逻辑注入。
统一使用 SDPA (Scaled Dot-Product Attention) 版本的注意力模块。

主要更新:
- [Fix] 输出类名修正为 LLaVAOneVision1_5_ModelOutputWithPast。
- [安全] 在传递 context 前严格检查接收模块是否为包装类。
- [逻辑] 完善全局剪枝与合并逻辑，强制使用 PruningCache。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List, Union
import traceback
import math

# 导入基础适配器类
from pruning.adapters.base_adapter import BasePruningAdapter
from pruning.pruning_strategies import get_pruning_strategy
from pruning.pruning_cache import PruningCache
from pruning.pruning_modules import get_context_key

# 从指定的本地源代码路径导入原始模型组件
from .source_code.modeling_llavaonevision1_5 import (
    LLaVAOneVision1_5_Model,
    LLaVAOneVision1_5_TextModel,
    LLaVAOneVision1_5_DecoderLayer,
    LLaVAOneVision1_5_SdpaAttention,
    RiceTransformerPretrainedModel,
    RiceBlock,
    RiceSdpaAttention,
    LLaVAOneVision1_5_ForConditionalGeneration,
    # === modify start: Correct Output Class ===
    LLaVAOneVision1_5_ModelOutputWithPast,
    # === modify end ===
    apply_rotary_pos_emb_vision,
    apply_rotary_pos_emb,
    repeat_kv
)

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast


# ================================================================================
# |                           1. 视觉塔 (Rice ViT) 包装类                        |
# ================================================================================

class Wrapped_RiceSdpaAttention(RiceSdpaAttention):
    """包装 Rice 视觉 SDPA 注意力模块，用于捕获注意力图和 Q/K 向量"""
    def __init__(self, original_attn: RiceSdpaAttention, pruning_conf: Dict[str, Any], layer_idx: int):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_attn.__dict__)
        self.pruning_config = pruning_conf
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # === modify start: add context ===
        context: Dict[str, Any] = None,
        # === modify end ===
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        
        # === modify start: capture pre-rope Q/K ===
        if context is not None:
            need_pre = self.pruning_config.get("needs", {}).get("need_vit_pre_rope_qk", False)
            q_key = get_context_key("vit_pre_rope_q", need_pre, "vit", self.layer_idx)
            k_key = get_context_key("vit_pre_rope_k", need_pre, "vit", self.layer_idx)
            if q_key: context[q_key] = q
            if k_key: context[k_key] = k
        # === modify end ===

        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # === modify start: capture post-rope Q/K ===
        if context is not None:
            need_post = self.pruning_config.get("needs", {}).get("need_vit_post_rope_qk", False)
            q_key = get_context_key("vit_post_rope_q", need_post, "vit", self.layer_idx)
            k_key = get_context_key("vit_post_rope_k", need_post, "vit", self.layer_idx)
            if q_key: context[q_key] = q
            if k_key: context[k_key] = k
        # === modify end ===

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # [Fix]: 获取 head_dim，因为 RiceSdpaAttention 可能没有 self.head_dim 属性
        head_dim = q.shape[-1]
        
        # === modify start: capture raw attention ===
        need_attn = self.pruning_config.get("needs", {}).get("need_vit_raw_attention", False)
        if context is not None and need_attn:
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
            attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            
            attn_key = get_context_key("vit_attn_map_list", need_attn, "vit", self.layer_idx)
            if attn_key: context[attn_key] = attn_weights
            
            attn_output = torch.matmul(attn_weights, v)
        else:
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
            attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v)
        # === modify end ===
        
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        
        # === modify start: capture cu_seqlens ===
        if context is not None:
            need_cu = self.pruning_config.get("needs", {}).get("need_cu_seqlens", False)
            cu_key = get_context_key("cu_seqlens", need_cu, "vit", self.layer_idx)
            if cu_key: context[cu_key] = cu_seqlens
        # === modify end ===
        
        return self.proj(attn_output)


class Prunable_RiceBlock(RiceBlock):
    """包装 Rice 视觉块，传递 Context"""
    def __init__(self, original_block: RiceBlock, pruning_conf: Dict[str, Any], layer_idx: int):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_block.__dict__)
        self.layer_idx = layer_idx
        self.pruning_config = pruning_conf
        
        if not isinstance(self.attn, Wrapped_RiceSdpaAttention):
            self.attn = Wrapped_RiceSdpaAttention(self.attn, pruning_conf, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # === modify start: add context ===
        context: Dict[str, Any] = None,
        # === modify end ===
    ) -> torch.Tensor:
        
        # === modify start: safe call with context ===
        if isinstance(self.attn, Wrapped_RiceSdpaAttention):
            attn_output = self.attn(
                self.norm1(hidden_states),
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                context=context
            )
        else:
            attn_output = self.attn(
                self.norm1(hidden_states),
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings
            )
        # === modify end ===

        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Prunable_RiceTransformer(RiceTransformerPretrainedModel):
    """包装 Rice 视觉塔主体，注入全局信息到 Context"""
    def __init__(self, original_visual: RiceTransformerPretrainedModel, pruning_config: Dict[str, Any]):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_visual.__dict__)
        self.pruning_config = pruning_config

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        grid_thw: torch.Tensor,
        # === modify start: add context ===
        context: Dict[str, Any] = None,
        # === modify end ===
    ) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        img_feats = hidden_states.shape[0]
        
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        cu = cu_seqlens.to(torch.long)
        num_segments = cu.numel() - 1
        cls_token = self.class_embedding.to(hidden_states.dtype).unsqueeze(0)

        total_patches = cu[-1].item()
        new_total = total_patches + num_segments
        D = hidden_states.size(-1)
        new_hidden = hidden_states.new_empty((new_total, D))
        new_rotary_pos_emb = rotary_pos_emb.new_empty((new_total, rotary_pos_emb.shape[-1]))

        write_ptr = 0
        new_cu = [0]
        for i in range(1, num_segments + 1):
            seg_start = cu[i-1].item()
            seg_end = cu[i].item()
            seg_len = seg_end - seg_start
            new_hidden[write_ptr] = cls_token
            new_rotary_pos_emb[write_ptr] = self.class_pos_emb
            new_hidden[write_ptr + 1: write_ptr + 1 + seg_len] = hidden_states[seg_start:seg_end]
            new_rotary_pos_emb[write_ptr + 1: write_ptr + 1 + seg_len] = rotary_pos_emb[seg_start:seg_end]
            write_ptr += 1 + seg_len
            new_cu.append(write_ptr)

        hidden_states = new_hidden
        cu_seqlens = torch.tensor(new_cu, device=hidden_states.device, dtype=torch.int32) 
        rotary_pos_emb = new_rotary_pos_emb

        hidden_states = self.pre_layernorm(hidden_states)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # === modify start: inject context info ===
        if context is not None:
            context["spatial_merge_size"] = self.spatial_merge_size
            context["cu_seqlens_full"] = cu_seqlens
        # === modify end ===

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                # === modify start: safe checkpointing ===
                if isinstance(blk, Prunable_RiceBlock):
                     hidden_states = self._gradient_checkpointing_func(
                        blk.__call__, hidden_states, cu_seqlens, None, position_embeddings,
                        context
                    )
                else:
                    hidden_states = self._gradient_checkpointing_func(
                        blk.__call__, hidden_states, cu_seqlens, None, position_embeddings
                    )
                # === modify end ===
            else:
                # === modify start: safe call ===
                if isinstance(blk, Prunable_RiceBlock):
                    hidden_states = blk(
                        hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings,
                        context=context
                    )
                else:
                    hidden_states = blk(
                        hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings
                    )
                # === modify end ===
        
        new_hidden = hidden_states.new_empty((img_feats, D))

        for i in range(1, num_segments + 1):
            seg_start = cu[i-1].item()
            seg_end = cu[i].item()
            new_hidden[seg_start:seg_end] = hidden_states[seg_start+1:seg_end+1]
        hidden_states = new_hidden

        return self.merger(hidden_states)


# ================================================================================
# |                         2. 语言模型 (LLM) 包装类                             |
# ================================================================================

class Wrapped_LLaVAOV_SdpaAttention(LLaVAOneVision1_5_SdpaAttention):
    """包装 LLM SDPA 注意力层，捕获 LLM 内部状态"""
    def __init__(self, original_attn: LLaVAOneVision1_5_SdpaAttention, pruning_conf: Dict[str, Any], layer_idx: int):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_attn.__dict__)
        self.pruning_config = pruning_conf
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # === modify start: add context ===
        context: Dict[str, Any] = None,
        # === modify end ===
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # === modify start: custom forward injection ===
        if context is not None and "custom_attention_forward_fns" in context:
            custom_fns = context["custom_attention_forward_fns"]
            if "Wrapped_LLaVAOV_SdpaAttention" in custom_fns:
                return custom_fns["Wrapped_LLaVAOV_SdpaAttention"](
                    self, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_value=past_key_value, output_attentions=output_attentions,
                    use_cache=use_cache, cache_position=cache_position,
                    position_embeddings=position_embeddings, context=context
                )
        # === modify end ===

        if output_attentions:
            return super().forward(
                hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # === modify start: capture pre-rope ===
        if context is not None:
            need_pre = self.pruning_config.get("needs", {}).get("need_pre_rope_qk", False)
            q_key = get_context_key("pre_rope_q", need_pre, "llm", self.layer_idx)
            k_key = get_context_key("pre_rope_k", need_pre, "llm", self.layer_idx)
            if q_key: context[q_key] = query_states
            if k_key: context[k_key] = key_states
        # === modify end ===

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # === modify start: capture post-rope and feature map ===
        if context is not None:
            need_post = self.pruning_config.get("needs", {}).get("need_post_rope_qk", False)
            q_key = get_context_key("post_rope_q", need_post, "llm", self.layer_idx)
            k_key = get_context_key("post_rope_k", need_post, "llm", self.layer_idx)
            if q_key: context[q_key] = query_states
            if k_key: context[k_key] = key_states
            
            need_feat = self.pruning_config.get("needs", {}).get("need_feature_map", False)
            f_key = get_context_key("feature_map", need_feat, "llm", self.layer_idx)
            if f_key: context[f_key] = hidden_states
        # === modify end ===

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and input_shape[1] > 1 else False

        # === modify start: capture attention map ===
        need_map = self.pruning_config.get("needs", {}).get("need_attn_map", False)
        if context is not None and need_map:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if causal_mask is not None: attn_weights = attn_weights + causal_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            map_key = get_context_key("attn_map", need_map, "llm", self.layer_idx)
            if map_key: context[map_key] = attn_weights
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0, is_causal=is_causal,
            )
        # === modify end ===

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class Prunable_LLaVAOV_DecoderLayer(LLaVAOneVision1_5_DecoderLayer):
    """包装 LLM 解码器层，实现层级剪枝"""
    def __init__(self, original_layer: LLaVAOneVision1_5_DecoderLayer, pruning_conf: Dict[str, Any], layer_idx: int):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_layer.__dict__)
        self.layer_idx = layer_idx
        
        if not isinstance(self.self_attn, Wrapped_LLaVAOV_SdpaAttention):
            self.self_attn = Wrapped_LLaVAOV_SdpaAttention(self.self_attn, pruning_conf, layer_idx)
        
        self.pruning_config = pruning_conf
        self.pruning_fn = get_pruning_strategy(pruning_conf.get("method")) if "method" in pruning_conf else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # === modify start: add context ===
        context: Dict[str, Any] = None,
        # === modify end ===
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # === modify start: custom layer forward injection ===
        if context is not None and "custom_layer_forward_fns" in context:
            custom_fns = context["custom_layer_forward_fns"]
            if "Prunable_LLaVAOV_DecoderLayer" in custom_fns:
                return custom_fns["Prunable_LLaVAOV_DecoderLayer"](
                    self, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_value=past_key_value, output_attentions=output_attentions,
                    use_cache=use_cache, cache_position=cache_position,
                    position_embeddings=position_embeddings, context=context, **kwargs
                )
        # === modify end ===

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # === modify start: safe call with context ===
        if isinstance(self.self_attn, Wrapped_LLaVAOV_SdpaAttention):
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                context=context
            )
        else:
             hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        # === modify end ===
        
        hidden_states = residual + hidden_states

        # === modify start: Layer-wise Pruning Logic ===
        if self.pruning_fn and isinstance(past_key_value, PruningCache) and past_key_value.is_prefill_stage(self.layer_idx, timing="after_update"):
            # [新增] 记录剪枝前的形状以计算统计数据
            shape_before = hidden_states.shape
            
            keep_mask = self.pruning_fn(context, **self.pruning_config.get("params", {}))
            if isinstance(keep_mask, torch.Tensor) and keep_mask.dtype == torch.bool:
                keep_mask = keep_mask.view(-1)
                hidden_states = hidden_states[:, keep_mask, :]
                context["keep_mask"] = keep_mask
                past_key_value.set_pruning_mask_for_layer(self.layer_idx, keep_mask)
                
                # [新增] 打印层级剪枝统计信息
                pruned_tokens = shape_before[1] - hidden_states.shape[1]
                print("=" * 50)
                print(f"[Layer Pruning] Layer {self.layer_idx}")
                print(f"Sequence length before: {shape_before[1]}")
                print(f"Sequence length after: {hidden_states.shape[1]}")
                print(f"Pruned tokens: {pruned_tokens}")
                print(f"Pruning method: {self.pruning_config.get('method', 'unknown')}")
                print("=" * 50)
        # === modify end ===

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class Prunable_LLaVAOV_TextModel(LLaVAOneVision1_5_TextModel):
    """包装 LLM 文本模型入口，负责全局剪枝、合并及状态同步"""
    def __init__(self, original_model: LLaVAOneVision1_5_TextModel, pruning_conf: Dict[str, Any]):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_model.__dict__)
        self.pruning_config = pruning_conf
        self.pruning_fn = get_pruning_strategy(pruning_conf.get("method")) if "method" in pruning_conf else None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # === modify start: add context ===
        context: Dict[str, Any] = None,
        # === modify end ===
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # === modify start: Force PruningCache ===
        if use_cache and not isinstance(past_key_values, PruningCache):
            past_key_values = PruningCache(config=self.config)
        # === modify end ===

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)

        # === modify start: inject context ===
        if context is not None:
            context.update({
                "inputs_embeds": inputs_embeds, 
                "text_pos_ids": position_ids, 
                "pos_emb_ids": position_ids.unsqueeze(0),
                "feature_map": inputs_embeds
            })
        # === modify end ===

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # === modify start: GLOBAL PRUNING / MERGING LOGIC (Prefill) ===
        if self.pruning_fn and isinstance(past_key_values, PruningCache) and past_key_values.is_prefill_stage(0, timing="before_update"):
            hidden_states_shape_before = hidden_states.shape
            keep_mask = self.pruning_fn(context, **self.pruning_config.get("params", {}))
            
            # Case A: 纯剪枝
            if isinstance(keep_mask, torch.Tensor) and keep_mask.dtype == torch.bool:
                mask_flat = keep_mask.view(-1)
                
                # 切片 Hidden States
                hidden_states = hidden_states[:, mask_flat, :]
                
                # [新增] 打印全局剪枝统计信息
                pruned_tokens_total = hidden_states_shape_before[1] - hidden_states.shape[1]
                print("=" * 50)
                print(f"[Global Pruning] Layer 0 (Prefill)")
                print(f"Sequence length before: {hidden_states_shape_before[1]}")
                print(f"Sequence length after: {hidden_states.shape[1]}")
                print(f"Pruned tokens: {pruned_tokens_total}")
                print(f"Pruning method: {self.pruning_config.get('method', 'unknown')}")
                print("=" * 50)
                
                # 切片 RoPE Embeddings (LLaVA-OV RoPE 格式为 (cos, sin))
                position_embeddings = (
                    position_embeddings[0][:, mask_flat, :], 
                    position_embeddings[1][:, mask_flat, :]
                )
                
                # 切片 Causal Mask
                if causal_mask is not None: 
                    # Mask Shape: [Batch, 1, Seq, Seq]
                    causal_mask = causal_mask[:, :, mask_flat, :][:, :, :, mask_flat]
                
                # 切片 Cache Position & IDs
                if cache_position is not None: 
                    cache_position = cache_position[mask_flat] 
                if position_ids is not None:
                    # position_ids [Batch, Seq]
                    position_ids = position_ids[:, mask_flat]

                # 切片 Context
                if "input_ids" in context: context["input_ids"] = context["input_ids"][:, mask_flat]
                if "pos_emb_ids" in context: context["pos_emb_ids"] = position_ids
                if "text_pos_ids" in context: context["text_pos_ids"] = position_ids[0]
                
                past_key_values.set_pruning_mask_for_layer("global", mask_flat)
                
            # Case B: 合并 (Merging)
            elif isinstance(keep_mask, tuple):
                (merge_weight_list, sizes_list_gpu, is_vision_list_gpu, merge_prun_mask) = keep_mask
                update_position_ids = self.pruning_config.get("params", {}).get("update_position_ids", True)
                
                hidden_list = []
                position_ids_list = []
                
                cur_img_id = 0
                sizes_list = sizes_list_gpu.tolist()
                
                hidden_split = torch.split(hidden_states, sizes_list, dim=1)
                pos_ids_split = torch.split(position_ids, sizes_list, dim=1)
                
                for hidden_part, pos_part, is_vision in zip(hidden_split, pos_ids_split, is_vision_list_gpu):
                    if is_vision:
                        # 视觉部分: 执行加权合并
                        merge_weight = merge_weight_list[cur_img_id].to(hidden_states.device).to(hidden_states.dtype)
                        if merge_weight.dim() == 2: merge_weight = merge_weight.unsqueeze(0)
                        cur_img_id += 1
                        
                        hidden_list.append(torch.matmul(merge_weight, hidden_part))
                        
                        if update_position_ids:
                            # 重新计算合并后的 Position IDs
                            pos_part_float = pos_part.float()
                            weight_t = merge_weight.transpose(1, 2)
                            merged_pos = torch.matmul(pos_part_float, weight_t)
                            position_ids_list.append(merged_pos)
                    else:
                        # 非视觉部分: 保持原样
                        hidden_list.append(hidden_part)
                        if update_position_ids:
                            position_ids_list.append(pos_part.float())
                
                hidden_states = torch.cat(hidden_list, dim=1)
                
                # [新增] 打印全局合并统计信息
                merged_tokens_total = hidden_states_shape_before[1] - hidden_states.shape[1]
                print("=" * 50)
                print(f"[Global Merging] Layer 0 (Prefill)")
                print(f"Sequence length before: {hidden_states_shape_before[1]}")
                print(f"Sequence length after: {hidden_states.shape[1]}")
                print(f"Merged tokens: {merged_tokens_total}")
                print(f"Merging method: {self.pruning_config.get('method', 'unknown')}")
                print("=" * 50)

                if update_position_ids:
                    position_ids = torch.cat(position_ids_list, dim=1).long()
                else:
                    position_ids = position_ids[:, merge_prun_mask]
                
                # 重新计算 RoPE (位置变了)
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                
                if causal_mask is not None:
                     causal_mask = causal_mask[:, :, merge_prun_mask, :][:, :, :, merge_prun_mask]
                
                if "input_ids" in context: context["input_ids"] = context["input_ids"][:, merge_prun_mask]
                if cache_position is not None: 
                    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
                
                past_key_values.set_pruning_mask_for_layer("global", merge_prun_mask)

        # Decode 阶段补偿
        if isinstance(past_key_values, PruningCache) and not past_key_values.is_prefill_stage(0, timing="before_update"):
            is_p, _, p_count = past_key_values.get_pruning_mask_for_layer("global")
            if is_p and p_count > 0:
                if cache_position is not None:
                    cache_position = cache_position - p_count
        # === modify end ===

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # === modify start: safe call with context ===
            if isinstance(decoder_layer, Prunable_LLaVAOV_DecoderLayer):
                layer_outputs = decoder_layer(
                    hidden_states, 
                    attention_mask=causal_mask, 
                    position_ids=position_ids,
                    past_key_value=past_key_values, 
                    output_attentions=output_attentions,
                    use_cache=use_cache, 
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    context=context
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            # === modify end ===

            hidden_states = layer_outputs[0]
            if use_cache: next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions: all_self_attns += (layer_outputs[1],)

            # === modify start: Layer-wise Sync ===
            if context is not None and "keep_mask" in context:
                k_mask = context.pop("keep_mask")
                
                if causal_mask is not None: 
                    causal_mask = causal_mask[:, :, k_mask, :][:, :, :, k_mask]
                
                if cache_position is not None: 
                    cache_position = cache_position[k_mask]
                
                if position_ids is not None:
                    position_ids = position_ids[:, k_mask]
                
                if position_embeddings is not None:
                     position_embeddings = (
                         position_embeddings[0][:, k_mask, :], 
                         position_embeddings[1][:, k_mask, :]
                     )

            if (
                not past_key_values.is_prefill_stage(layer_idx, timing="after_update")
                and past_key_values.get_pruning_mask_for_layer(layer_idx)[0]
            ):
                _, keep_mask, pruned_tokens = (
                    past_key_values.get_pruning_mask_for_layer(layer_idx)
                )
                if (
                    causal_mask is not None
                    and causal_mask.shape[-1]
                    > keep_mask.shape[-1]
                ):
                    padding_shape = (
                        causal_mask.shape[-1]
                        - keep_mask.shape[-1]
                    )
                    padding_true = torch.ones(
                        padding_shape, dtype=keep_mask.dtype, device=keep_mask.device
                    )
                    effective_mask = torch.cat([keep_mask, padding_true], dim=-1)
                else:
                    effective_mask = keep_mask
                if causal_mask is not None:
                    causal_mask = causal_mask[:, :, :, effective_mask]
                if cache_position is not None:
                    cache_position = cache_position - pruned_tokens
            # === modify end ===

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ================================================================================
# |                        3. 顶层 LlavaOV 模型包装器                             |
# ================================================================================

class Prunable_LLaVAOV_Model(LLaVAOneVision1_5_Model):
    """包装顶层 LlavaModel"""
    def __init__(self, original_model: LLaVAOneVision1_5_Model, pruning_config: Dict[str, Any]):
        torch.nn.Module.__init__(self)
        self.__dict__.update(original_model.__dict__)
        self.pruning_config = pruning_config

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
        cache_position: Optional[torch.LongTensor] = None,
        # === modify start: add context ===
        context: Dict[str, Any] = None,
        # === modify end ===
    ) -> Union[Tuple, LLaVAOneVision1_5_ModelOutputWithPast]:
        # === modify start: init context ===
        if context is None: context = {}
        context.update({"model_config": self.config, "input_ids": input_ids, "image_grid_thw": image_grid_thw})
        # === modify end ===

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                # === modify start: pass context to get_image_features ===
                # 注意：get_image_features 返回 Tensor，不再是 Tuple
                image_embeds = self.get_image_features(pixel_values, image_grid_thw, context=context)
                # === modify end ===
                
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                     raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                
                # === modify start: Inline mask calculation (Restore original logic) ===
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                
                # Capture vision mask for pruning strategies
                context["vision_token_mask"] = image_mask[0, :, 0]
                # === modify end ===

            if pixel_values_videos is not None:
                # === modify start: pass context to get_video_features ===
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, context=context)
                # === modify end ===
                
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                # === modify start: Inline mask calculation (Restore original logic) ===
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                if "vision_token_mask" in context:
                     context["vision_token_mask"] = context["vision_token_mask"] | video_mask[0, :, 0]
                else:
                     context["vision_token_mask"] = video_mask[0, :, 0]
                # === modify end ===

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # === modify start: safe call with context ===
        if isinstance(self.language_model, Prunable_LLaVAOV_TextModel):
            outputs = self.language_model(
                input_ids=None, position_ids=position_ids, attention_mask=attention_mask,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=True,
                cache_position=cache_position,
                context=context
            )
        else:
             outputs = self.language_model(
                input_ids=None, position_ids=position_ids, attention_mask=attention_mask,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=True,
                cache_position=cache_position
            )
        # === modify end ===

        output = LLaVAOneVision1_5_ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return output if return_dict else output.to_tuple()

    # === modify start: override feature methods to pass context ===
    def get_image_features(self, pixel_values, image_grid_thw, context=None):
        pixel_values = pixel_values.type(self.visual.dtype)
        # Safe call for visual
        if isinstance(self.visual, Prunable_RiceTransformer):
             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw, context=context)
        else:
             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        
        # [Fix]: 直接返回 Tensor，不要 split
        return image_embeds

    def get_video_features(self, pixel_values_videos, video_grid_thw, context=None):
        return self.get_image_features(pixel_values_videos, video_grid_thw, context=context)
    # === modify end ===


# ================================================================================
# |                           4. 适配器核心类实现                                |
# ================================================================================

class LlavaOVPruningAdapter(BasePruningAdapter):
    """针对 LLaVA OneVision 1.5 模型的剪枝适配器"""

    def wrap_model(self) -> torch.nn.Module:
        """执行模块替换逻辑"""
        print("Activating LLaVA-OneVision-1.5 Pruning Plugin via Adapter...")
        model = self.model
        config = self.config
        wrapped_components = set()

        if "vl_model" in config and not isinstance(model.model, Prunable_LLaVAOV_Model):
            model.old_model["model"] = model.model
            model.model = Prunable_LLaVAOV_Model(model.model, config.get("vl_model", {}))
            wrapped_components.add("vl_model")

        if "vision_model" in config and not isinstance(model.model.visual, Prunable_RiceTransformer):
            model.old_model["visual"] = model.model.visual
            model.model.visual = Prunable_RiceTransformer(model.model.visual, config.get("vision_model", {}))
            wrapped_components.add("vision_model")

        if "text_model" in config and not isinstance(model.model.language_model, Prunable_LLaVAOV_TextModel):
            model.old_model["language_model"] = model.model.language_model
            model.model.language_model = Prunable_LLaVAOV_TextModel(model.model.language_model, config.get("text_model", {}))
            wrapped_components.add("text_model")

        if "vision_blocks" in config:
            if "vision_blocks" not in model.old_model:
                model.old_model["vision_blocks"] = {}
            for idx_str, p_conf in config["vision_blocks"].items():
                idx = int(idx_str)
                orig_block = model.model.visual.blocks[idx]
                if not isinstance(orig_block, Prunable_RiceBlock):
                    model.old_model["vision_blocks"][idx] = {"block": orig_block, "attn": orig_block.attn}
                    model.model.visual.blocks[idx] = Prunable_RiceBlock(orig_block, p_conf, idx)
                    wrapped_components.add(f"vision_block_{idx}")

        if "decoder_layers" in config:
            if "decoder_layers" not in model.old_model:
                model.old_model["decoder_layers"] = {}
            for idx_str, p_conf in config["decoder_layers"].items():
                idx = int(idx_str)
                orig_layer = model.model.language_model.layers[idx]
                if not isinstance(orig_layer, Prunable_LLaVAOV_DecoderLayer):
                    model.old_model["decoder_layers"][idx] = {"layer": orig_layer, "self_attn": orig_layer.self_attn}
                    model.model.language_model.layers[idx] = Prunable_LLaVAOV_DecoderLayer(orig_layer, p_conf, idx)
                    wrapped_components.add(f"decoder_layer_{idx}")

        print(f"Wrapping verified for: {wrapped_components}")
        return model

    def unwrap_model(self) -> torch.nn.Module:
        """执行模块还原逻辑"""
        print("Recovering original LLaVA-OneVision-1.5 model...")
        model = self.model
        if hasattr(model, "old_model") and model.old_model:
            old = model.old_model

            if "decoder_layers" in old:
                for idx, info in old["decoder_layers"].items():
                    model.model.language_model.layers[idx] = info["layer"]
                    model.model.language_model.layers[idx].self_attn = info["self_attn"]

            if "vision_blocks" in old:
                for idx, info in old["vision_blocks"].items():
                    model.model.visual.blocks[idx] = info["block"]
                    model.model.visual.blocks[idx].attn = info["attn"]

            if "language_model" in old:
                model.model.language_model = old["language_model"]
            if "visual" in old:
                model.model.visual = old["visual"]
            if "model" in old:
                model.model = old["model"]

            if hasattr(model, "_pruning_adapter"):
                del model._pruning_adapter
            
            model.old_model = {}
            print("✅ Model successfully restored.")
        return model