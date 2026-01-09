# -*- coding: utf-8 -*-
"""
================================================================================
|                    适配器抽象基类 (pruning/adapters/base_adapter.py)                     |
================================================================================
文件功能:
本文件定义了 `BasePruningAdapter` 抽象基类。它使用 Python 的 `abc` 模块来
创建一个接口规范，强制所有具体的模型适配器（如 Qwen2.5-VL 适配器）都必须
实现 `wrap_model` 和 `unwrap_model` 这两个核心方法。

设计目的:
- **规范接口**: 确保所有适配器都提供统一的模块替换和恢复功能。
- **强制实现**: 防止在创建新适配器时遗漏关键功能。
- **多态性**: 允许 `pruning_modules.py` 中的工厂函数以统一的方式处理任何
  类型的适配器对象，无需关心其内部的具体实现。
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any


class BasePruningAdapter(ABC):
    """
    所有模型剪枝适配器的抽象基类。
    """

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        """
        初始化适配器。

        Args:
            model (torch.nn.Module): 要被修改的原始模型实例。
            config (Dict[str, Any]): 描述剪枝行为的配置字典。
        """
        self.model = model
        self.config = config
        if not hasattr(self.model, "old_model"):
            self.model.old_model = {}

    @abstractmethod
    def wrap_model(self) -> torch.nn.Module:
        """
        根据配置，用可剪枝的模块包装原始模型。
        必须实现此方法。

        Returns:
            torch.nn.Module: 被修改后的模型。
        """
        pass

    @abstractmethod
    def unwrap_model(self) -> torch.nn.Module:
        """
        将模型恢复到其原始状态。
        必须实现此方法。

        Returns:
            torch.nn.Module: 恢复后的原始模型。
        """
        pass
