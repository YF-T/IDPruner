# -*- coding: utf-8 -*-
#
# pruning/configs/models.py
#
"""
================================================================================
|       模型配置注册表 (pruning/configs/models.py) v2.2 (Qwen3 Update)       |
================================================================================
文件功能:
本文件是整个剪枝与分析框架的“模型注册表”，作为所有支持模型的配置信息的
“单一事实来源”(Single Source of Truth)。它被放置在 `pruning` 目录下，
强调了模型定义是剪枝框架自身的核心配置之一。

v2.2 更新:
- [模型新增] 添加了 Qwen3-VL-2B-Instruct 和 Qwen3-VL-32B-Instruct 的配置。
- [配置更新] 更新了所有 Qwen3-VL 模型的 `layers` 字段为 `["0"]`。
- [配置更新] 确认并更新了所有 Qwen3-VL 模型的 `vision_config.deepstack_visual_indexes`
  和 `vision_config.depth`。
- [v2.1 保留] 将所有模型的字典键名（key）统一为不含 'Qwen/' 前缀的格式。
- [v2.1 保留] 将所有模型的 'path' 字段统一设置为 Hugging Face Hub ID。
"""
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    LlavaForConditionalGeneration,  # <-- 新增导入
    AutoProcessor,
)

try:
    from pruning.adapters.source_code.modeling_llavaonevision1_5 import (
        LLaVAOneVision1_5_ForConditionalGeneration,
    )
except ImportError:
    print("Warning: LLaVAOneVision1_5_ForConditionalGeneration could not be imported from local source.")
    import traceback
    print(traceback.format_exc())
    LLaVAOneVision1_5_ForConditionalGeneration = None

MODEL_CONFIGS = {
    # --- Qwen2.5-VL 系列 ---
    "Qwen2.5-VL-3B-Instruct": {
        "path": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_class": Qwen2_5_VLForConditionalGeneration,
        "processor_class": AutoProcessor,
        "pruning_adapter_path": "pruning.adapters.qwen2_5_vl_adapter.Qwen2_5_VLPruningAdapter",
        "attribute_paths": {
            "llm_layers": "model.language_model.layers",
            "vision_blocks": "model.visual.blocks",
        },
        "forward_param_map": {},
        "layers": ["0", "2", "6", "10", "15", "20"],  # Qwen2.5 保持不变
        "total_llm_layers": 36,
        "vision_config": {
            "fullatt_block_indexes": [7, 15, 23, 31],
            "patch_size": 14,
            "spatial_merge_size": 2,
            "depth": 32,  # 添加 Qwen2.5 的 depth
        },
    },
    "Qwen2.5-VL-7B-Instruct": {
        "path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_class": Qwen2_5_VLForConditionalGeneration,
        "processor_class": AutoProcessor,
        "pruning_adapter_path": "pruning.adapters.qwen2_5_vl_adapter.Qwen2_5_VLPruningAdapter",
        "attribute_paths": {
            "llm_layers": "model.language_model.layers",
            "vision_blocks": "model.visual.blocks",
        },
        "forward_param_map": {},
        "layers": ["0", "2", "6", "10", "15", "20"],  # Qwen2.5 保持不变
        "total_llm_layers": 28,
        "vision_config": {
            "fullatt_block_indexes": [7, 15, 23, 31],
            "patch_size": 14,
            "spatial_merge_size": 2,
            "depth": 32,  # 添加 Qwen2.5 的 depth
        },
    },
    # --- LLaVA 系列 ---
    "llava-1.5-7b-hf": {
        "path": "llava-hf/llava-1.5-7b-hf",
        "model_class": LlavaForConditionalGeneration,
        "processor_class": AutoProcessor,
        "pruning_adapter_path": "pruning.adapters.llava_adapter.LlavaPruningAdapter",
        "attribute_paths": {
            "llm_layers": "model.language_model.layers",
            "vision_blocks": "model.vision_tower.vision_model.encoder.layers",
        },
        "forward_param_map": {},
        "layers": ["0"],  # 默认在第0层进行 Global Pruning
        "total_llm_layers": 32,  # Llama-2-7b based
        "vision_config": {
            # LLaVA 默认使用 feature layer -2 (对应索引 22)
            # heuristic_methods.py 使用 fullatt_block_indexes[-1] 来确定 last_global_layer
            "fullatt_block_indexes": [22],
            "patch_size": 14,
            "depth": 24,  # CLIP-ViT-L-336
        },
    },
    "LLaVA-OneVision-1.5-8B-Instruct": {
        "path": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        "model_class": LLaVAOneVision1_5_ForConditionalGeneration,
        "processor_class": AutoProcessor,
        "pruning_adapter_path": "pruning.adapters.llava_ov_1_5_adapter.LlavaOVPruningAdapter",
        "attribute_paths": {
            "llm_layers": "model.language_model.layers",
            "vision_blocks": "model.visual.blocks",
        },
        "forward_param_map": {},
        "layers": ["0"],
        "total_llm_layers": 36,
        "vision_config": {
            # 这里的配置来自 config.json
            "depth": 24,
            "patch_size": 14,
            "spatial_merge_size": 2,
            # 用于自动推断 last_global_layer
            "fullatt_block_indexes": [23],
        },
    },
}
