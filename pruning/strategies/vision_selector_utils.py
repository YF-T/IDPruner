# -*- coding: utf-8 -*-
"""
================================================================================
|     通用 Vision Selector 分数提取工具 (`vision_selector_utils.py`) v2.0     |
================================================================================
文件功能:
1. 提供万能的 Vision Selector (TransformerScorer) 分数提取接口。
2. 兼容 V1 (单一视觉输入, 如 forward(x)) 和 V2 (视觉+文本输入, 如 forward(v, t)) 架构。
3. 采用“单例注册表”机制，确保模型加载、源码内省 (inspect) 等耗时操作仅执行一次。
4. 核心计算逻辑：支持与训练配置一致的 Soft Top-K (Differentiable Top-K) 权重计算。

设计原则:
- 宁愿因架构不匹配报错，也不产生模糊的预测结果。
- 自动从 config.json 匹配 __init__ 参数，实现无缝初始化。
"""

import torch
import torch.nn as nn
import os
import json
import importlib.util
import time
import glob
import inspect
from typing import Dict, Any, Tuple, Optional, List

# 逻辑概念到代码参数名的候选映射表
# 用于在内省 (inspect) 时自动识别模型 forward 函数中各个参数的物理含义
CONCEPT_MAPPING = {
    "vision": ["vision_hidden", "x", "hidden_states", "visual_embeds", "xs", "visual_hidden", "vision_embeds"],
    "text": ["text_hidden", "text_embeds", "context_hidden", "textual_hidden", "text_features"]
}

# 全局单例注册表：存储已初始化的模型实例及其元数据
# 结构: { selector_path: { "model": nn.Module, "arg_mapping": {param_name: role}, "dtype": dtype } }
_SELECTOR_REGISTRY: Dict[str, Dict[str, Any]] = {}

# ================================================================================
# |                      1. Soft Top-K (微分 Top-K) 核心计算                     |
# ================================================================================

@torch.no_grad()
def _find_ts(xs: torch.Tensor, k: float) -> torch.Tensor:
    """
    使用二分查找求解偏移量 ts，使得 sum(sigmoid(xs + ts)) ≈ k。
     xs: [B, N] 原始分数
     k: 目标保留的 Token 绝对数量
    """
    # 确保在 float32 下进行数值计算
    xs = xs.float()
    
    # 设定搜索边界
    lo = -xs.max(dim=1, keepdims=True).values - 10.0
    hi = -xs.min(dim=1, keepdims=True).values + 10.0
    
    # 执行 64 次迭代以获得足够精度
    for _ in range(64):
        mid = (hi + lo) / 2
        # 计算当前的软数量 (Soft Count)
        current_k = torch.sigmoid(xs + mid).sum(dim=1)
        mask = current_k < k
        lo[mask] = mid[mask]
        hi[~mask] = mid[~mask]
        
    ts = (lo + hi) / 2
    return ts

def compute_soft_topk_weight(scores: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
    """
    将原始分数转换为 Soft Top-K 权重。
    返回: sigmoid(scores + ts)，结果在 [0, 1] 之间。
    注意：根据要求，ratio 默认固定为 0.2。
    """
    orig_dtype = scores.dtype
    xs = scores.float()
    b, n = xs.shape
    
    target_k = n * ratio
    # 极小/极大值保护
    target_k = max(1e-4, min(target_k, n - 1e-4))
    
    ts = _find_ts(xs, target_k)
    return torch.sigmoid(xs + ts).to(dtype=orig_dtype)

# ================================================================================
# |                      2. 自动化适配与模型加载逻辑                             |
# ================================================================================

def _init_selector_entry(selector_path: str, device: torch.device):
    """
    执行一次性的初始化工作：
    1. 加载源码 2. 探测构造函数签名并初始化 3. 探测 forward 签名建立映射 4. 加载权重
    """
    if not os.path.exists(selector_path):
        raise FileNotFoundError(f"❌ Vision Selector 路径不存在: {selector_path}")

    # 1. 自动定位关键文件
    config_path = os.path.join(selector_path, "config.json")
    weights_files = glob.glob(os.path.join(selector_path, "*.safetensors")) + \
                    glob.glob(os.path.join(selector_path, "*.bin"))
    py_files = glob.glob(os.path.join(selector_path, "*.py"))

    if not weights_files or not py_files:
        raise FileNotFoundError(f"❌ Selector 目录组件不完整 (需 .py 定义和权重文件): {selector_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 2. 动态加载类定义
    module_name = f"dynamic_selector_mod_{int(time.time()*1000)}"
    spec = importlib.util.spec_from_file_location(module_name, py_files[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    # 识别类名 (优先从 config 读取 architecture 字段)
    arch_name = config.get("architecture", "TransformerScorer")
    ModelClass = getattr(mod, arch_name, None) or getattr(mod, "TransformerScorer", None)
    if not ModelClass:
        raise AttributeError(f"❌ 在 {py_files[0]} 中找不到对应的 Scorer 类定义。")

    # 3. 自动化参数对齐: 初始化 (__init__)
    # 获取构造函数所需的参数列表 (剔除 self, args, kwargs)
    init_sig = inspect.signature(ModelClass.__init__)
    required_init_params = [
        p for p in init_sig.parameters.keys() 
        if p not in ['self', 'args', 'kwargs']
    ]
    
    # 从 config.json 中自动提取匹配的参数
    init_kwargs = {p: config[p] for p in required_init_params if p in config}
    model = ModelClass(**init_kwargs)

    # 4. 自动化参数对齐: 执行 (forward) - 确定 Vision/Text 对应关系
    forward_sig = inspect.signature(model.forward)
    forward_params = [p for p in forward_sig.parameters.keys() if p != 'self']
    
    arg_mapping = {} # 代码参数名 -> 逻辑概念 (vision/text)
    for p_name in forward_params:
        if p_name in CONCEPT_MAPPING["vision"]:
            arg_mapping[p_name] = "vision"
        elif p_name in CONCEPT_MAPPING["text"]:
            arg_mapping[p_name] = "text"
            
    # 检查基本兼容性
    if "vision" not in arg_mapping.values():
        raise TypeError(f"❌ 无法识别模型 {arch_name}.forward 的视觉输入参数名。可用候选: {CONCEPT_MAPPING['vision']}")

    # 5. 加载权重并清洗前缀
    try:
        from safetensors.torch import load_file
        state_dict = load_file(weights_files[0]) if weights_files[0].endswith(".safetensors") else torch.load(weights_files[0], map_location="cpu")
    except Exception:
        state_dict = torch.load(weights_files[0], map_location="cpu")
    
    # 移除训练框架产生的冗余前缀 (module., visual.importance_scorer. 等)
    cleaned_sd = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "").replace("visual.importance_scorer.", "").replace("_orig_mod.", "")
        cleaned_sd[new_k] = v
        
    model.load_state_dict(cleaned_sd, strict=False)
    model.to(device).eval()

    # 6. 写入单例注册表缓存
    _SELECTOR_REGISTRY[selector_path] = {
        "model": model,
        "arg_mapping": arg_mapping,
        "dtype": next(model.parameters()).dtype
    }

# ================================================================================
# |                      3. 公共接口函数 (对外唯一入口)                          |
# ================================================================================

def get_universal_selector_scores(
    selector_path: str,
    vision_hidden: torch.Tensor,
    text_hidden: Optional[torch.Tensor] = None,
    mode: str = "soft_topk",  # 可选: "soft_topk", "softmax", "raw"
    **extra_kwargs
) -> torch.Tensor:
    """
    通用分数提取接口。自动处理缓存、探测、参数对齐及后处理。
    
    参数:
        selector_path: 模型文件夹路径。
        vision_hidden: 视觉特征 [B, N, D]。
        text_hidden:   文本特征 [B, M, D] (V2 架构必需，V1 架构自动忽略)。
        mode:          分数后处理模式。
                       - "soft_topk": 返回 [0, 1] 权重 (ratio 固定为 0.2)。
                       - "softmax":   返回归一化概率。
                       - "raw":       返回模型原始输出。
        extra_kwargs:  支持透传 position_ids 等额外参数（如果模型定义支持）。
        
    返回:
        归一化或原始的重要性分数 [B, N]。
    """
    # 1. 确保模型已加载至缓存
    if selector_path not in _SELECTOR_REGISTRY:
        _init_selector_entry(selector_path, vision_hidden.device)
    
    entry = _SELECTOR_REGISTRY[selector_path]
    model = entry["model"]
    dtype = entry["dtype"]

    # 2. 极简执行路径：根据映射表组装参数
    call_params = {}
    for arg_name, role in entry["arg_mapping"].items():
        if role == "vision":
            call_params[arg_name] = vision_hidden.to(dtype=dtype)
        elif role == "text":
            if text_hidden is not None:
                call_params[arg_name] = text_hidden.to(dtype=dtype)
            else:
                # 若模型需要文本但未提供，则该参数留空，由模型内部默认值或后续报错处理
                pass
            
    # 透传额外参数
    for k, v in extra_kwargs.items():
        call_params[k] = v

    # 3. 推理阶段
    with torch.no_grad():
        outputs = model(**call_params)
        # 统一输出形状为 [B, N]
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)

    # 4. 后处理分发
    if mode == "soft_topk":
        # 强制执行训练时的 0.2 比例约束
        return compute_soft_topk_weight(outputs, ratio=0.2)
    elif mode == "softmax":
        return torch.softmax(outputs.float(), dim=-1).to(dtype=outputs.dtype)
    else:
        return outputs