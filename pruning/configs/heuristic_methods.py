# -*- coding: utf-8 -*-
"""
================================================================================
|       启发式剪枝方法定义与配置生成 (`heuristic_methods.py`) v1.7       |
================================================================================
文件功能:
本文件包含:
1.  HEURISTIC_METHOD_DEFINITIONS: 一个字典，集中定义所有启发式剪枝方法的
    核心属性（名称、策略函数、作用域、参数类型、基础参数、依赖、注册键、显示模板）。
    这是方法定义的“单一事实来源”。
2.  generate_pruning_method_configs: 一个函数，用于为 `run_analysis.py`
    动态生成特定层和比例/步长下的启发式方法配置字典。此版本动态读取
    HEURISTIC_METHOD_DEFINITIONS 来生成配置，输出为简洁的新版格式。

v1.7 更新:
- [重构] `generate_pruning_method_configs` 函数现在动态地遍历
  `HEURISTIC_METHOD_DEFINITIONS` 来生成所有方法的配置，移除了硬编码逻辑。
- [格式统一] 输出配置保持简洁格式，包含 `name` 键，不含顶层冗余键。
- [健壮性] 增加了对 stride 方法在 ratio 无法匹配时的跳过处理。
- [v1.6 保留] [逻辑恢复] `generate_pruning_method_configs` 函数现在使用用户提供的旧版逻辑
  来决定包含哪些方法及其参数。(此条在v1.7中被覆盖)
- [v1.5 保留] [格式恢复] 恢复了 `generate_pruning_method_configs` 生成配置中冗余的顶层
  `method`, `params`, `needs` 键，以严格匹配用户提供的旧版函数输出格式。(此条在v1.7中被覆盖)
- [v1.4 保留] [Bug修复] 为 `HEURISTIC_METHOD_DEFINITIONS` 中的每个方法添加了
  `registration_key` 字段，其值对应 `_PRUNING_STRATEGIES` 中的注册键。
- [v1.4 保留] [Bug修复] 修改 `generate_pruning_method_configs` 函数，使其在生成的配置中
  使用正确的 `registration_key` 作为 "method" 字段的值，解决了 `ValueError`。(此条在v1.7中被覆盖)
- [v1.2 保留] 修复了因尝试导入不存在的 `MODEL_VISION_CONFIGS` 导致的 `ImportError`。
- [v1.2 保留] 修改 `generate_pruning_method_configs` 函数以从 `MODEL_CONFIGS` 获取视觉配置。
- [v1.2 保留] 为 Qwen3 模型添加了初步的适配逻辑（使用 deepstack_visual_indexes）并增加了警告。
"""
from typing import Dict, Any
import numpy as np  # 需要 numpy 用于 isclose

# 从同目录下的 models.py 导入模型配置
from .models import MODEL_CONFIGS

# ================================================================================
# |            **启发式方法核心定义 (HEURISTIC_METHOD_DEFINITIONS)** |
# ================================================================================

HEURISTIC_METHOD_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # --- [新添加] Baseline (不剪枝) ---
    "baseline": {
        "name": "Baseline (No Pruning)",
        "method_fn": "baseline_pruning",  # 仅供参考
        "registration_key": "baseline",  # <-- 必须与 pruning_strategies.py 中的键匹配
        "scope": "llm",  # 作用域设为 "llm"
        "type": "ratio",  # 设为 "ratio" 使其能被 generate 函数拾取
        "base_params": {},
        "needs": {},
        "display_name_template": "Baseline (R={ratio:.2f})",  # 名字会带 ratio，但不影响功能
    },
    # --- LLM Scope Methods ---
    "random": {
        "name": "Random",
        "method_fn": "random_pruning",  # 实际函数名 (仅供参考)
        "registration_key": "random",  # <-- 用于 get_pruning_strategy
        "scope": "llm",
        "type": "ratio",
        "base_params": {},
        "needs": {},
        "display_name_template": "Random (R={ratio:.2f})",
    },
    "fastv": {
        "name": "FastV",
        "method_fn": "special_token_based_attention_pruning",
        "registration_key": "special_token_based_attention",  # <-- 用于 get_pruning_strategy
        "scope": "llm",
        "type": "ratio",
        "base_params": {
            "use_post_rope": True,
            "query_source": {"strategy": "last_text"},
        },
        "needs": {"need_post_rope_qk": True},
        "display_name_template": "Post-Attn-LastText (R={ratio:.2f})",
    },
    "divprune": {
        "name": "DivPrune",
        "method_fn": "divprune",
        "registration_key": "divprune",  # <-- 用于 get_pruning_strategy
        "scope": "vit",
        "type": "ratio",
        "base_params": {},
        "needs": {"need_feature_map": True},
        "display_name_template": "Diversity (R={ratio:.2f})",
    },
    "dart": {
        "name": "DART",
        "method_fn": "dart_pruning",
        "registration_key": "dart",  # <-- 用于 get_pruning_strategy
        "scope": "llm",
        "type": "ratio",
        "base_params": {
            "pivot_image_token": 4,
            "pivot_text_token": 4,
            "use_post_rope": True,
        },
        "needs": {"need_feature_map": True, "need_post_rope_qk": True},
        "display_name_template": "DART(4,4) (R={ratio:.2f})",
    },
    "visionzip": {
        "name": "VisionZip",
        "method_fn": "vision_zip_merging",  # 仅作文档说明，实际调用由 registration_key 决定
        "registration_key": "vision_zip_merging",  # 必须与 pruning/merging_strategies.py 中的注册名一致
        "scope": "vit",
        "type": "ratio",
        "base_params": {
            "zip_ratio": 10.0 / 64.0,  # 按照论文里的 54 : 10 比例设置
            "update_position_ids": False,  # <--- 在这里配置
        },
        "dynamic_params": ["layer_idx"],  # 自动填充为 ViT 的最后一层索引
        "needs": {
            "need_vit_post_rope_qk": "specific",  # 需要指定层的 Post-RoPE Q/K 用于重计算
            "need_image_grid_thw": True,  # 需要图像网格信息来拆分 Batch
        },
        "display_name_template": "VisionZip (R={ratio:.2f})",
    },
    "vispruner": {
        "name": "VisPruner",
        "method_fn": "vispruner_pruning",
        "registration_key": "vispruner", # 对应 pruning_strategies.py 中的注册键
        "scope": "vit",
        "type": "ratio",
        "base_params": {
            "important_ratio_of_kept": 0.5, # 默认重要性 Token 占比
        },
        "dynamic_params": ["layer_idx"], # 关键：告诉生成器自动注入 last_global_layer
        "needs": {
            "need_vit_post_rope_qk": "specific", # 阶段一必需
            "need_feature_map": True             # 阶段二必需
        },
        "display_name_template": "VisPruner (R={ratio:.2f})",
    },
    "hiprune": {
        "name": "HiPrune",
        "method_fn": "hiprune_pruning",
        "registration_key": "hiprune",
        "scope": "vit",
        "type": "ratio",
        "base_params": {
            "alpha": 0.1,  # Default alpha
            "recompute_attention": True, # HiPrune 需要精确的 Q/K 交互
        },
        "model_related_base_params": {
            # --- LLaVA-1.5 (Object Layer = 9) ---
            "llava-1.5-7b-hf": {"object_layer": 9},
            
            # --- Qwen2.5-VL (Object Layer = 16) ---
            "Qwen2.5-VL-3B-Instruct": {"object_layer": 16},
            "Qwen2.5-VL-7B-Instruct": {"object_layer": 16},

            # --- OneVision LLaVA-1.5 (Object Layer = 16) ---
            "LLaVA-OneVision-1.5-8B-Instruct": {"object_layer": 16},
        },
        "dynamic_params": ["last_vit_layer"], # 自动获取最后一层 ViT 索引
        "needs": {
            "need_vit_post_rope_qk": "specific", # 需要特定层的 Q/K 用于计算分数
            "need_image_grid_thw": True, # Qwen 计算宽度需要
        },
        "display_name_template": "HiPrune(R={ratio:.2f})",
    },
    "scope": {
        "name": "SCOPE",
        "method_fn": "scope_pruning",
        "registration_key": "scope",
        "scope": "vit",
        "type": "ratio",
        "base_params": {
            "alpha": 1.0,        # 显著性缩放因子 (显式参数)
            "combined": "multi", # 融合模式 (环境变量控制)
        },
        "dynamic_params": ["layer_idx"], # 自动填充为视觉塔最后一层
        "needs": {
            "need_feature_map": True,
            "need_vit_post_rope_qk": "specific",
        },
        "display_name_template": "SCOPE (R={ratio:.2f})",
    },
    "vision_selector": {
        "name": "VisionSelector (origin)",
        "method_fn": "vision_selector_pruning",
        "registration_key": "vision_selector",
        "scope": "vit",
        "type": "ratio",
        "base_params": {

        },
        "model_related_base_params": {
            "Qwen2.5-VL-7B-Instruct": {
                "selector_path": "pretrained_selector/Extracted_Selector_Qwen2.5-VL-7B-official",
            },
            "Qwen2.5-VL-3B-Instruct": {
                "selector_path": "pretrained_selector/Extracted_Selector_Qwen2.5-VL-3B-official",
            },
            "llava-1.5-7b-hf": {
                "selector_path": "pretrained_selector/Extracted_Selector_Llava1.5-7B",
            },
            "LLaVA-OneVision-1.5-8B-Instruct": {
                "selector_path": "pretrained_selector/Extracted_Selector_LLaVA-OV-1.5-8B-official",
            },
        },
        "needs": {},
        # [修改] 增加 Origin 后缀以示区分
        "display_name_template": "VS-Origin (R={ratio:.2f})",
    },
}

# [新增/修改] 动态生成并行化 MMR 配置
# 扫描 Lambda 参数 (0.0 到 1.0) 和 并行度 K
HEURISTIC_METHOD_DEFINITIONS.update(
    {
        f"idpruner_lambda{lam}": {
            "name": f"IDPruner (Lambda={lam})",
            "method_fn": "idpruner",
            "registration_key": "idpruner",
            "scope": "vit",
            "type": "ratio",
            "base_params": {
                "mmr_lambda": lam,
                "parallel_k": 1, # 注入并行度参数
            },
            "model_related_base_params": {
                "Qwen2.5-VL-7B-Instruct": {
                    "selector_path": "pretrained_selector/Extracted_Selector_Qwen2.5-VL-7B-official",
                },
                "Qwen2.5-VL-3B-Instruct": {
                    "selector_path": "pretrained_selector/Extracted_Selector_Qwen2.5-VL-3B-official",
                },
                "llava-1.5-7b-hf": {
                    "selector_path": "pretrained_selector/Extracted_Selector_Llava1.5-7B",
                },
                "LLaVA-OneVision-1.5-8B-Instruct": {
                    "selector_path": "pretrained_selector/Extracted_Selector_LLaVA-OV-1.5-8B-official",
                },
            },
            "needs": {
                "need_inputs_embeds": True
            },
            # 模板中加入 k 以便在报告中区分
            "display_name_template": f"IDPruner (Lambda={lam})",
        }
        for lam in [0.1, 0.3, 0.5, 0.7, 0.9] # 示例 Lambda 扫描
    }
)

# ================================================================================
# |            配置生成函数 (为 run_analysis.py 保留的兼容接口)            |
# ================================================================================


def generate_pruning_method_configs(model_name: str, layer: str, ratio: float) -> Dict:
    """
    为所有启发式剪枝方法动态生成配置字典。

    [修改逻辑]:
    根据方法定义中的 `requires_full_llm_wrapping` Flag 来决定是否生成全层配置。
    """
    configs = {}

    # 1. 获取模型特定的配置信息
    if model_name not in MODEL_CONFIGS:
        print(
            f"警告: 在 models.py 中未找到模型 '{model_name}' 的配置。ViT相关方法可能无法生成。"
        )
        vision_config = {}
        last_global_layer = -1
        total_llm_layers = 0
    else:
        model_conf = MODEL_CONFIGS[model_name]
        vision_config = model_conf.get("vision_config", {})
        total_llm_layers = model_conf.get("total_llm_layers", 0)  # 获取 LLM 总层数

        fullatt_indexes = vision_config.get("fullatt_block_indexes")
        deepstack_indexes = vision_config.get("deepstack_visual_indexes")
        last_global_layer = -1
        if fullatt_indexes:
            last_global_layer = fullatt_indexes[-1]
        elif deepstack_indexes:
            last_global_layer = deepstack_indexes[-1]
        else:
            print(f"警告 ({model_name}): 未找到 ViT 层索引。")

    # 遍历 HEURISTIC_METHOD_DEFINITIONS 动态生成配置
    for method_key, definition in HEURISTIC_METHOD_DEFINITIONS.items():
        # print(f"处理方法: {method_key}")
        scope = definition["scope"]
        method_type = definition["type"]
        base_params = definition["base_params"].copy()

        # === [核心修改]: 处理模型相关参数 (model_related_base_params) ===
        model_related = definition.get("model_related_base_params", {})
        find_flag = False
        for model_key, params in model_related.items():
            if model_name == model_key:
                base_params.update(params)
                find_flag = True
                break
        # if not find_flag and model_related and model_related != {}:
        #     raise ValueError(f"未找到模型 {model_name} 的相关参数在方法 {method_key} 中。")
        # =============================================================

        needs = definition["needs"]
        vit_needs = definition.get("vit_needs", {})
        registration_key = definition["registration_key"]
        display_name_template = definition["display_name_template"]

        # [新增] 获取 Flag，默认为 False
        requires_full_wrapping = definition.get("requires_full_llm_wrapping", False)

        final_params = {}
        final_config = {}
        display_name = ""

        # --- [通用逻辑] 根据 Flag 构建 decoder_layers 字典 ---
        decoder_layers_config = {}

        # 如果 Flag 为 True，或者 scope 为 'llm' (LLM方法通常针对特定层，但也需要基础结构)，
        # 我们需要根据情况处理。
        # 这里的逻辑是：
        # 1. 如果 requires_full_wrapping=True: 生成 {0:{}, 1:{}, ...} 全层空字典。
        # 2. 如果 scope='llm': 只需要目标层 {layer: {...}}，通常不需要全层 wrap (除非 stride 等特殊需求)。
        #    但为了逻辑清晰，我们保持原有的 'llm' 处理方式 (只 wrap 目标层)，
        #    仅当 requires_full_wrapping=True 时才强制全层。

        if requires_full_wrapping:
            if total_llm_layers > 0:
                # 生成全层空字典，强制 Adapter 替换所有 Attention
                decoder_layers_config = {str(i): {} for i in range(total_llm_layers)}
                # print(f"生成全层空字典: {decoder_layers_config}")
            else:
                # 无法获取总层数，无法生成全层配置
                print(
                    f"警告: 模型 {model_name} 未定义 total_llm_layers，但方法 {method_key} 请求了 full_wrapping。"
                )

        if scope == "llm":
            if total_llm_layers == 0:
                continue

            should_generate = False
            if method_type == "ratio":
                final_params = {**base_params, "ratio": ratio}
                display_name = display_name_template.format(ratio=ratio)
                should_generate = True
            elif method_type == "stride":
                s = 0
                if np.isclose(ratio, 0.75):
                    s = 2
                elif np.isclose(ratio, 8 / 9):
                    s = 3
                elif np.isclose(ratio, 15 / 16):
                    s = 4

                if s > 0:
                    bias_val = base_params.get("bias", 0)
                    if bias_val == -1:
                        bias_val = s - 1
                    final_params = {"stride": s, "bias": bias_val}
                    display_name = display_name_template.format(stride=s, bias=bias_val)
                    should_generate = True

            if should_generate:
                # 对于 LLM 方法，我们在 decoder_layers_config 中填入目标层的具体参数
                # 注意：如果 requires_full_wrapping=True，这里会覆盖掉字典中该层的空字典
                # 如果 requires_full_wrapping=False，decoder_layers_config 初始为空，这里添加一项
                decoder_layers_config[str(layer)] = {
                    "method": registration_key,
                    "params": final_params,
                    "needs": needs,
                }

                final_config = {
                    "vl_model": {},
                    "text_model": {} if not vit_needs else {"needs": vit_needs},
                    "decoder_layers": decoder_layers_config,
                    "name": display_name,
                }
                configs[display_name] = final_config

        elif scope == "vit":
            if last_global_layer == -1:
                continue

            if method_type == "ratio":
                final_params = {**base_params, "ratio": ratio}
                # 处理动态参数
                dynamic_params_list = definition.get("dynamic_params", [])
                if "layer_idx" in dynamic_params_list:
                    final_params["layer_idx"] = last_global_layer
                if "last_vit_layer" in dynamic_params_list:
                    final_params["last_vit_layer"] = last_global_layer
                object_layer = base_params.get("object_layer")

                # 构建 vision_blocks 配置
                vision_blocks_config = {}
                layers_needing_capture = set()
                if "last_vit_layer" in final_params or "layer_idx" in final_params:
                    layers_needing_capture.add(last_global_layer)
                if object_layer is not None:
                    layers_needing_capture.add(object_layer)
                for vit_layer_idx in layers_needing_capture:
                    vision_blocks_config[str(vit_layer_idx)] = {"needs": needs}

                display_name = display_name_template.format(ratio=ratio)

                final_config = {
                    "vl_model": {},
                    "vision_model": {} if not vit_needs else {"needs": vit_needs},
                    "vision_blocks": vision_blocks_config,
                    "text_model": {
                        "method": registration_key,
                        "params": final_params,
                        "needs": needs,
                    },
                    # [关键] 传入构建好的 decoder_layers_config
                    # 如果 requires_full_wrapping=True，这里包含了所有层的空字典
                    # 如果 False，这里是空字典
                    "decoder_layers": decoder_layers_config,
                    "name": display_name,
                }
                configs[display_name] = final_config

    return configs
