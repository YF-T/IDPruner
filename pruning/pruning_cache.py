# -*- coding: utf-8 -*-
"""
Custom Cache for Pruning

This file defines a custom cache class that inherits from transformers.DynamicCache
and adds functionalities for managing pruning state during generation.
"""
from typing import Optional, Dict, Any, List
import torch
from transformers.cache_utils import DynamicCache
from transformers import PretrainedConfig


class PruningCache(DynamicCache):
    """
    A custom cache that extends DynamicCache to manage pruning information.

    This cache adds two main functionalities:
    1.  Determines whether the current forward pass is in the 'prefill' or 'decode' stage for each layer,
        which is crucial for deciding when to apply pruning.
    2.  Stores pruning masks for each layer in a dedicated attribute `pruning_info`.
    """

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
        **kwargs
    ):
        super().__init__(
            config=config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
            **kwargs
        )

        # 1. 存储剪枝信息
        # 结构: {layer_idx: {"pruned": True/False, "mask": tensor(...)}}
        self.pruning_info: Dict[int, Dict[str, Any]] = {}

        # 2. 按层记录update的调用次数
        self.update_counts: Dict[int, int] = {}

    def is_prefill_stage(self, layer_idx: int, timing: str = "after_update") -> bool:
        """
        Checks if the current stage for a given layer is prefill or decoding.

        Args:
            layer_idx (int): The index of the layer to check.
            timing (str, optional): The point in time for the check.
                - 'before_update': Checks the state before the current layer's cache is updated.
                - 'after_update': Checks the state after the current layer's cache has been updated.
                Defaults to 'after_update'.

        Returns:
            bool: True if it's the prefill stage, False otherwise.
        """
        count = self.update_counts.get(layer_idx, 0)

        if timing == "before_update":
            # 在第一次update调用前，计数为0，是prefill阶段
            return count == 0
        elif timing == "after_update":
            # 在第一次update调用后，计数为1，是prefill阶段
            return count == 1
        else:
            raise ValueError("timing must be either 'before_update' or 'after_update'")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Overrides the base update method to increment the update counter for the layer.
        """
        # First, call the original update logic
        updated_key, updated_value = super().update(
            key_states, value_states, layer_idx, cache_kwargs
        )

        # Then, increment the counter for this layer
        self.update_counts[layer_idx] = self.update_counts.get(layer_idx, 0) + 1

        return updated_key, updated_value

    def set_pruning_mask_for_layer(self, layer_idx: int, mask: torch.Tensor):
        """
        Stores the pruning mask for a specific layer.
        This would be called from within a PrunableBlock/Layer after pruning.
        """
        assert mask.dim() == 1, "Pruning mask should be a 1D tensor."
        self.pruning_info[layer_idx] = {
            "pruned": True,
            "mask": mask,
            "pruned_tokens": mask.size(0) - mask.sum().item(),
        }

    def get_pruning_mask_for_layer(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieves the pruning mask for a specific layer.
        """
        return (
            (layer_idx in self.pruning_info),
            self.pruning_info.get(layer_idx, {}).get("mask"),
            self.pruning_info.get(layer_idx, {}).get("pruned_tokens", 0),
        )

    def reset(self):
        """
        Resets the cache, pruning information, and update counts for a new generation.
        """
        super().reset()
        self.pruning_info = {}
        self.update_counts = {}
