# -*- coding: utf-8 -*-
"""
================================================================================
|       通用剪枝插件工厂模块 (pruning/pruning_modules.py) v2.1       |
================================================================================
文件功能:
本文件是剪枝插件的通用“工厂”，负责动态地为任何已注册的模型激活或恢复剪枝功能。
它本身不包含任何模型专属的逻辑，而是通过适配器模式（Adapter Pattern）将
模型相关的实现完全解耦。

v2.1 更新:
- [健壮性增强] _get_adapter_class 函数现在支持回退查找。如果使用模型的完整路径
  （如 'Qwen/Qwen2.5-VL-7B-Instruct'）无法在配置字典中找到键，它会自动尝试
  使用路径的最后一部分（如 'Qwen2.5-VL-7B-Instruct'）再次查找，以兼容
  transformers 库保存的 name_or_path 属性。

核心函数:
- enable_pruning:
  - 识别传入的模型类型。
  - 从 `configs/models.py` 中查找该模型的配置，特别是 `pruning_adapter_path`。
  - 动态导入并实例化该模型专属的适配器类。
  - 调用适配器的 `wrap_model()` 方法来执行实际的模块替换。
- recover_original_model:
  - 从模型对象中获取之前存储的适配器实例。
  - 调用适配器的 `unwrap_model()` 方法来恢复所有被替换的模块。

设计模式:
- **工厂模式 (Factory Pattern)**: `enable_pruning` 根据输入动态创建并返回一个
  被修改过的产品（模型）。
- **适配器模式 (Adapter Pattern)**: 每个模型家族的专属逻辑被封装在独立的
  适配器类中，为通用工厂提供了统一的接口。
"""

from typing import Dict, Any, Optional
import torch
import importlib
from transformers import PreTrainedModel
from pruning.configs.models import MODEL_CONFIGS


def get_context_key(
    base_key: str, need_value: Any, module_type: str, layer_idx: int
) -> Optional[str]:
    """
    根据配置为context字典生成一个唯一的键。
    """
    if need_value is True:
        return base_key
    elif need_value == "specific":
        return f"{base_key}_{module_type}_{layer_idx}"
    else:
        return None


def _get_adapter_class(model: PreTrainedModel):
    """动态导入并返回模型专属的适配器类。"""
    model_name_or_path = model.config.name_or_path

    # 步骤 1: 尝试直接使用 name_or_path 查找
    model_conf = MODEL_CONFIGS.get(model_name_or_path)

    # 步骤 2: 如果直接查找失败，尝试使用路径的最后一部分（短名称）作为键
    short_model_name = None
    if model_conf is None:
        # 兼容 Hugging Face Hub ID (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
        # 或本地路径 (e.g., "/path/to/Qwen2.5-VL-7B-Instruct")
        short_model_name = model_name_or_path.split("/")[-1]
        model_conf = MODEL_CONFIGS.get(short_model_name)

    # 步骤 3: 如果两次尝试都失败，则抛出错误
    if not model_conf:
        error_msg = (
            f"在 `configs/models.py` 中未找到模型 '{model_name_or_path}' 的配置。"
        )
        if short_model_name and short_model_name != model_name_or_path:
            error_msg += f" 同样也未找到其短名称 '{short_model_name}' 的配置。"
        raise ValueError(error_msg)

    adapter_path_str = model_conf.get("pruning_adapter_path")
    if not adapter_path_str:
        raise ValueError(
            f"模型 '{model_name_or_path}' 的配置中缺少 `pruning_adapter_path` 字段。"
        )

    try:
        module_path, class_name = adapter_path_str.rsplit(".", 1)
        adapter_module = importlib.import_module(module_path)
        AdapterClass = getattr(adapter_module, class_name)
        return AdapterClass
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"无法从路径 '{adapter_path_str}' 动态导入适配器类: {e}")


def enable_pruning(model: PreTrainedModel, config: Dict) -> PreTrainedModel:
    """
    通过动态加载的适配器，激活模型的剪枝功能。

    Args:
        model (PreTrainedModel): 待修改的Hugging Face模型实例。
        config (Dict): 描述剪枝行为的配置字典。

    Returns:
        PreTrainedModel: 被修改后的、具备剪枝能力的模型实例。
    """
    print("=" * 50)
    print("通用剪枝工厂: 正在激活插件...")
    print("=" * 50)

    AdapterClass = _get_adapter_class(model)
    adapter = AdapterClass(model, config)

    # 将adapter实例附加到模型上，以便恢复时使用
    model._pruning_adapter = adapter

    # 调用适配器的wrap方法执行实际的模块替换
    wrapped_model = adapter.wrap_model()

    return wrapped_model


def recover_original_model(model: PreTrainedModel):
    """
    通过模型上附加的适配器，将模型恢复到其原始状态。

    Args:
        model (PreTrainedModel): 之前被 enable_pruning 修改过的模型实例。
    """
    print("=" * 50)
    print("通用剪枝工厂: 正在恢复原始模型...")
    print("=" * 50)

    if hasattr(model, "_pruning_adapter") and model._pruning_adapter is not None:
        adapter = model._pruning_adapter
        adapter.unwrap_model()
        print("✅ 模型已成功恢复。")
    else:
        print("⚠️ 警告: 未在模型上找到活动的剪枝适配器，无需恢复。")
