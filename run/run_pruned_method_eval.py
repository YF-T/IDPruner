# -*- coding: utf-8 -*-
"""
================================================================================
|       多方法剪枝评测脚本 (`run_pruned_method_eval.py`) v2.2 (Auto Config)      |
================================================================================
文件功能:
本脚本使用 `lmms-eval` 框架，系统性地评测多种预定义的启发式剪枝方法在**单个指定模型**
上的性能表现。脚本为**串行执行**。

v2.2 更新:
- [核心修复] 配置生成逻辑不再硬编码，而是直接调用 `pruning.configs.heuristic_methods.generate_pruning_method_configs`。
  这确保了 `requires_full_llm_wrapping` 等高级配置标志能被正确处理（例如用于 Scale 注入）。
- [调试] 生成配置时会打印实际的 Config 字典内容。
- [保持] 包含 Base Model Debug 逻辑。

使用方法:
    python run/run_pruned_method_eval.py --models [MODEL_NAME] [其他可选参数]
"""
import os
import sys
import lmms_eval
from lmms_eval import evaluator
import torch
import json
import time
import traceback
from loguru import logger as eval_logger
import argparse
import copy
from collections import defaultdict
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# --- 动态添加项目根目录 ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

# --- 导入所需模块 ---
from pruning.pruning_modules import enable_pruning, recover_original_model
from utils.lmms_eval_qwen2_5_vl_pruned import Qwen2_5_VLPruned

# [关键导入] 导入配置定义和生成函数
from pruning.configs.heuristic_methods import (
    HEURISTIC_METHOD_DEFINITIONS,
    generate_pruning_method_configs,
)
from pruning.configs.models import MODEL_CONFIGS
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

# ==================== 1. 全局配置变量 (默认值) ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
eval_logger.info(f"使用设备: {DEVICE}")

# 实验方法参数 (默认值)
DEFAULT_PRUNING_RATIOS = [0.75, 0.9]
DEFAULT_PRUNING_STRIDES = [2, 3]

# 评估任务和参数 (默认值)
DEFAULT_TASKS = ["gqa", "textvqa", "scienceqa"]
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_FEWSHOT = 0
DEFAULT_OUTPUT_PATH = "./results"
DEFAULT_LOG_SAMPLES = True
DEFAULT_PREVIOUS_RESULTS_PATH = None
DEFAULT_MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
DEFAULT_LLM_PRUNING_LAYER = "0"
# =======================================================


# ==================== 理论加速比计算 (保持不变) ====================
def calculate_theoretical_speedup(
    total_layers: int, pruning_config: dict, meta: dict = None
) -> Dict[str, Any]:
    """
    计算给定剪枝配置的理论加速比。
    """
    cost_ratio = 1.0
    speedup_factor = 1.0
    note = ""

    # 处理 baseline 特殊情况
    if meta and meta.get("method_name") == "Baseline_debug":
        return {
            "cost_ratio": 1.0,
            "speedup_factor": 1.0,
            "note": "Baseline_debug - no pruning",
        }

    # Case 1: Per-layer pruning defined in decoder_layers
    if "decoder_layers" in pruning_config and pruning_config["decoder_layers"]:
        decoder_layers_config = pruning_config["decoder_layers"]
        # 过滤掉空配置（可能由 requires_full_wrapping 生成）
        active_layers = {
            k: v for k, v in decoder_layers_config.items() if "params" in v
        }

        if not active_layers:
            # 如果全是空配置（即只是占位符），视为不剪枝，或者检查 text_model
            if (
                "text_model" in pruning_config
                and "method" in pruning_config["text_model"]
            ):
                pass  # 转到 Case 2 处理
            else:
                return {
                    "cost_ratio": 1.0,
                    "speedup_factor": 1.0,
                    "note": "No active pruning params found",
                }
        else:
            first_layer_key = list(active_layers.keys())[0]
            first_layer_conf = active_layers[first_layer_key]

            if first_layer_conf.get("method") == "predictor_driven":
                ratio_param = first_layer_conf.get("params", {}).get("ratio")
                if ratio_param is not None:
                    cost_ratio = 1.0 - ratio_param
                    speedup_factor = (
                        1.0 / cost_ratio if cost_ratio > 0 else float("inf")
                    )
                    note = "Estimated based on ratio for predictor"
                else:
                    return {
                        "cost_ratio": "N/A",
                        "speedup_factor": "N/A",
                        "note": "Predictor ratio unknown",
                    }
            else:
                pruning_points = []
                for k, v in active_layers.items():
                    params = v.get("params", {})
                    layer_ratio = params.get("ratio")
                    if layer_ratio is None and "stride" in params:
                        stride_val = params["stride"]
                        layer_ratio = 1 - (1 / stride_val**2) if stride_val > 0 else 1.0
                    if layer_ratio is not None:
                        pruning_points.append((int(k), layer_ratio))

                if not pruning_points:
                    # 可能回落到 text_model
                    pass
                else:
                    pruning_points.sort()
                    remaining_computation = 0.0
                    current_token_ratio = 1.0
                    last_layer_processed = -1

                    for layer_idx, prune_ratio in pruning_points:
                        num_layers_in_segment = layer_idx - last_layer_processed
                        remaining_computation += (
                            num_layers_in_segment * current_token_ratio
                        )
                        current_token_ratio *= 1 - prune_ratio
                        last_layer_processed = layer_idx

                    num_layers_in_final_segment = total_layers - (
                        last_layer_processed + 1
                    )
                    remaining_computation += (
                        num_layers_in_final_segment * current_token_ratio
                    )

                    cost_ratio = (
                        remaining_computation / total_layers
                        if total_layers > 0
                        else 0.0
                    )
                    speedup_factor = (
                        1.0 / cost_ratio if cost_ratio > 0 else float("inf")
                    )
                    return {
                        "cost_ratio": round(cost_ratio, 4),
                        "speedup_factor": round(speedup_factor, 2),
                        "note": note,
                    }

    # Case 2: Global pruning defined in text_model (for ViT methods)
    if "text_model" in pruning_config and "method" in pruning_config["text_model"]:
        params = pruning_config["text_model"].get("params", {})
        ratio = params.get("ratio")
        if ratio is None and "stride" in params:
            stride_val = params["stride"]
            ratio = 1 - (1 / stride_val**2) if stride_val > 0 else 1.0

        if ratio is not None:
            cost_ratio = 1.0 - ratio
            speedup_factor = 1.0 / cost_ratio if cost_ratio > 0 else float("inf")
            note = "Estimated based on ratio for ViT method"
        else:
            return {
                "cost_ratio": "N/A",
                "speedup_factor": "N/A",
                "note": "Ratio/Stride not found for ViT method.",
            }

    return {
        "cost_ratio": round(cost_ratio, 4) if isinstance(cost_ratio, float) else "N/A",
        "speedup_factor": (
            round(speedup_factor, 2) if isinstance(speedup_factor, float) else "N/A"
        ),
        "note": note,
    }


# ==========================================================


# ==================== 重构后的 main 函数 ====================
def main():
    """主函数：解析参数，生成配置，加载模型一次，然后串行执行评测"""
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="多方法剪枝评测脚本 (v2.2 Auto-Gen Config)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"要评测的模型名称 (默认: {DEFAULT_MODEL_NAME})。",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="LLM剪枝层索引列表 (默认: 模型配置中指定)。",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=None,
        help=f"剪枝率列表 (默认: {DEFAULT_PRUNING_RATIOS})。",
    )
    parser.add_argument(
        "--strides",
        nargs="+",
        type=int,
        default=None,
        help=f"步长列表 (默认: {DEFAULT_PRUNING_STRIDES})。",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="要评测的方法 key 列表 (默认: 所有)。",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help=f"评测任务列表 (默认: {DEFAULT_TASKS})。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"结果输出目录 (默认: {DEFAULT_OUTPUT_PATH})。",
    )
    parser.add_argument(
        "--log_samples",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LOG_SAMPLES,
        help="是否记录样本。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="批处理大小 (当前仅支持 1)。",
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=DEFAULT_NUM_FEWSHOT, help="Few-shot 数量。"
    )
    args = parser.parse_args()

    # --- 参数校验 ---
    if args.batch_size != 1:
        eval_logger.warning("当前脚本仅稳定支持 batch_size=1，已强制设置为 1。")
        args.batch_size = 1
    if args.ratios and args.strides:
        eval_logger.error("错误: 不能同时指定 --ratios 和 --strides。")
        sys.exit(1)
    if args.models not in MODEL_CONFIGS:
        eval_logger.error(f"错误: 指定的模型 '{args.models}' 不在配置中。")
        sys.exit(1)

    model_name = args.models
    effective_ratios = args.ratios if args.ratios else DEFAULT_PRUNING_RATIOS
    effective_strides = args.strides if args.strides else DEFAULT_PRUNING_STRIDES
    effective_methods = args.methods
    effective_layers = args.layers

    # --- run_params 生成 ---
    run_params = []
    if args.ratios:
        eval_logger.info(f"使用命令行指定的 Ratios: {effective_ratios}")
        for r in effective_ratios:
            s_float = 1 / np.sqrt(1 - r) if r < 1 else float("inf")
            s_int = int(round(s_float))
            # 只有当 ratio 刚好对应一个整数 stride 时才设置 stride，否则为 None (Ratio-based)
            if s_int > 0 and np.isclose(r, 1 - 1 / s_int**2):
                run_params.append({"ratio": r, "stride": s_int})
            else:
                run_params.append({"ratio": r, "stride": None})
    elif args.strides:
        eval_logger.info(f"使用命令行指定的 Strides: {effective_strides}")
        for s in effective_strides:
            if s <= 0:
                continue
            r = 1 - (1 / s**2)
            run_params.append({"ratio": r, "stride": s})
    else:  # 默认
        eval_logger.info("使用默认 Ratios 和 Strides 组合。")
        seen_ratios = set()
        for r in DEFAULT_PRUNING_RATIOS:
            s_float = 1 / np.sqrt(1 - r) if r < 1 else float("inf")
            s_int = int(round(s_float))
            if s_int > 0 and np.isclose(r, 1 - 1 / s_int**2):
                run_params.append({"ratio": r, "stride": s_int})
            else:
                run_params.append({"ratio": r, "stride": None})
            seen_ratios.add(r)
        for s in DEFAULT_PRUNING_STRIDES:
            if s <= 0:
                continue
            r = 1 - (1 / s**2)
            if r not in seen_ratios:
                run_params.append({"ratio": r, "stride": s})
                seen_ratios.add(r)

    # --- 初始化 ---
    os.makedirs(args.output_dir, exist_ok=True)
    results_log = {}
    task_manager = lmms_eval.tasks.TaskManager()
    ALL_PRUNING_CONFIGS = {}

    # --- 打印生效参数 ---
    eval_logger.info("=" * 30 + " 生效的评测参数 " + "=" * 30)
    eval_logger.info(f"Model: {model_name}")
    eval_logger.info(
        f"LLM Layers: {effective_layers if effective_layers else 'All Defined for Model'}"
    )
    eval_logger.info(
        f"Methods: {effective_methods if effective_methods else 'All Defined'}"
    )
    eval_logger.info("=" * 80)

    # --- 生成实验配置 (调用 generate_pruning_method_configs) ---
    eval_logger.info(f"Generating experiment configurations for: {model_name}...")
    model_config_full = MODEL_CONFIGS[model_name]
    available_layers = model_config_full.get("layers", [])

    layers_to_run_for_model = []
    if effective_layers is None:
        layers_to_run_for_model = available_layers
    else:
        layers_to_run_for_model = [l for l in effective_layers if l in available_layers]

    # 如果 layers_to_run_for_model 为空（例如只测试 ViT 范围方法），确保循环至少执行一次以生成 ViT 配置
    iteration_layers = layers_to_run_for_model if layers_to_run_for_model else ["0"]

    methods_to_generate = (
        HEURISTIC_METHOD_DEFINITIONS.keys()
        if effective_methods is None
        else effective_methods
    )

    # 1. 遍历参数组合
    for param_combo in run_params:
        ratio = param_combo["ratio"]
        stride = param_combo["stride"]

        # 记录已处理的 ViT 方法，防止在层循环中重复添加
        processed_vit_methods_for_this_ratio = set()

        # 2. 遍历层
        for layer in iteration_layers:
            # === [核心] 调用集中式配置生成函数 ===
            # 这会返回当前 (model, layer, ratio) 组合下所有可能的启发式方法配置
            # 格式: { "Display Name": config_dict }
            generated_configs_map = generate_pruning_method_configs(
                model_name, layer, ratio
            )

            # 3. 筛选并提取配置
            for method_key in methods_to_generate:
                if method_key not in HEURISTIC_METHOD_DEFINITIONS:
                    continue

                definition = HEURISTIC_METHOD_DEFINITIONS[method_key]
                scope = definition["scope"]
                method_type = definition["type"]
                display_name_template = definition["display_name_template"]

                # 检查 stride 约束
                if method_type == "stride":
                    if stride is None:
                        continue
                    if not np.isclose(ratio, 1 - 1 / stride**2):
                        continue

                # 重构预期名称以匹配生成结果
                expected_display_name = ""
                bias_val = definition["base_params"].get("bias", 0)

                if method_type == "ratio":
                    expected_display_name = display_name_template.format(ratio=ratio)
                elif method_type == "stride":
                    if bias_val == -1:
                        bias_val = stride - 1
                    expected_display_name = display_name_template.format(
                        stride=stride, bias=bias_val
                    )

                # 如果生成器返回了这个配置
                if expected_display_name in generated_configs_map:
                    # ViT 方法去重
                    if scope == "vit":
                        combo_id = f"{method_key}_{ratio}"
                        if combo_id in processed_vit_methods_for_this_ratio:
                            continue
                        processed_vit_methods_for_this_ratio.add(combo_id)

                        unique_key = (
                            f"{model_name}_method_{method_key}_ratio_{ratio:.3f}"
                        )
                        if method_type == "stride":
                            unique_key += f"_stride_{stride}"
                    else:
                        unique_key = f"{model_name}_layer_{layer}_method_{method_key}_ratio_{ratio:.3f}"
                        if method_type == "stride":
                            unique_key += f"_stride_{stride}_bias_{bias_val}"

                    final_config = generated_configs_map[expected_display_name]

                    # 手动重建 Meta 信息 (用于日志记录和结果保存)
                    meta = {
                        "model_name": model_name,
                        "pruning_layer": layer if scope == "llm" else "N/A",
                        "method_name": definition["name"],
                        "ratio": ratio,
                    }
                    if stride is not None:
                        meta["stride"] = stride
                    if method_type == "stride":
                        meta["bias"] = bias_val

                    ALL_PRUNING_CONFIGS[unique_key] = {
                        "config": final_config,
                        "meta": meta,
                    }

    # --- 注入 Base Model Debug 配置 ---
    # base_model_key = f"{model_name}_base_model"
    # ALL_PRUNING_CONFIGS[base_model_key] = {
    #     "config": {},
    #     "meta": {"model_name": model_name, "method_name": "Baseline_debug", "pruning_layer": "N/A", "ratio": 0.0}
    # }

    eval_logger.info(f"Generated {len(ALL_PRUNING_CONFIGS)} configurations.")

    # --- 打印待执行配置 ---
    print("\n" + "=" * 30 + " 待执行的实验配置 " + "=" * 30)
    config_keys_sorted = sorted(
        ALL_PRUNING_CONFIGS.keys(),
        key=lambda x: (0, x) if "base_model" in x else (1, x),
    )
    for config_key in config_keys_sorted:
        print(f"- {config_key}")
        # --- [新增] 打印实际配置 ---
        cfg_to_print = ALL_PRUNING_CONFIGS[config_key]["config"]
        # 为了不刷屏，只打印关键部分，或者缩进打印
        # print(f"  Config: {json.dumps(cfg_to_print, indent=2, default=str)}")
    print("=" * 80 + "\n")

    # --- 加载模型 (一次性) ---
    eval_logger.info(f"Loading base model: {model_name}...")
    try:
        model_path = model_config_full["path"]
        ModelClass = model_config_full["model_class"]
        effective_device_map = DEVICE if DEVICE == "cpu" else "auto"

        loaded_model = ModelClass.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if DEVICE != "cpu" else torch.float32,
            attn_implementation="sdpa",
            device_map=effective_device_map,
            trust_remote_code=True,
        ).eval()

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        eval_logger.success("Base model loaded.")
    except Exception as e:
        eval_logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    live_results_path = os.path.join(
        args.output_dir,
        f"live_results_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )

    # --- 主评测循环 ---
    total_scheduled = len(ALL_PRUNING_CONFIGS)
    processed_count = 0

    for key in config_keys_sorted:
        processed_count += 1
        combo_data = ALL_PRUNING_CONFIGS[key]
        pruning_config = combo_data["config"]
        metadata = combo_data["meta"]
        is_baseline_case = metadata.get("method_name") == "Baseline_debug"

        eval_logger.info(
            f"--- Experiment {processed_count}/{total_scheduled}: {key} ---"
        )

        # --- [新增] 打印当前应用的配置 ---
        print(
            f"DEBUG: Applied Config for {key}:\n{json.dumps(pruning_config, indent=2, default=str)}"
        )

        try:
            # 计算加速比
            total_layers = model_config_full.get("total_llm_layers", -1)
            acceleration_info = calculate_theoretical_speedup(
                total_layers, pruning_config, meta=metadata
            )

            model_to_eval = None
            if is_baseline_case:
                recover_original_model(loaded_model)
                model_to_eval = loaded_model
            else:
                recover_original_model(loaded_model)
                pruned_model = enable_pruning(loaded_model, config=pruning_config)
                model_to_eval = pruned_model

            lmm_obj = Qwen2_5_VLPruned(
                model_instance=model_to_eval,
                processor_instance=processor,
                tokenizer_instance=tokenizer,
                device=DEVICE,
                batch_size=args.batch_size,
            )

            results = evaluator.simple_evaluate(
                model=lmm_obj,
                tasks=args.tasks,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                log_samples=args.log_samples,
                task_manager=task_manager,
            )

            results_log[key] = {
                "results": results["results"],
                "method_name": metadata.get("method_name"),
                "model_params": metadata,
                "pruning_config": pruning_config,
                "acceleration_info": acceleration_info,
            }

            with open(live_results_path, "w", encoding="utf-8") as f:
                json.dump(results_log, f, indent=4, ensure_ascii=False, default=str)

        except Exception as e:
            eval_logger.error(f"Error in {key}: {e}")
            traceback.print_exc()
        finally:
            if not is_baseline_case:
                recover_original_model(loaded_model)
            if "lmm_obj" in locals():
                del lmm_obj
            torch.cuda.empty_cache()

    # --- 最终保存 ---
    final_path = os.path.join(
        args.output_dir,
        f"final_results_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=4, ensure_ascii=False, default=str)
    eval_logger.info(f"Done. Saved to {final_path}")


if __name__ == "__main__":
    try:
        eval_logger.remove()
    except:
        pass
    eval_logger.add(sys.stderr, level="INFO")
    main()
