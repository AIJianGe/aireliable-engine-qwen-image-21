"""GGUF 加载后处理：非 Linear 字节张量还原（本轮实际修复的 bug）。

背景（2026-10-14 实测）：diffusers 0.41.0 ``from_single_file`` 把 Qwen-Image-2.1 GGUF
的 nn.Linear 换成 GGUFLinear（forward 时按 quant_type 反量化，正确），但**非 Linear
模块**（txt_in.text_norm 的 ZeroCenterRMSNorm，BF16 权重）拿到的是字节布局张量
（(4096,) BF16 → uint8(8192,)），前向 ``hidden(4096) * weight(8192)`` 抛 RuntimeError。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from gguf import GGMLQuantizationType

from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter

from engines.qwen_image_21_feeb8130.backends.gguf_backend import _repair_gguf_params


class _Norm(nn.Module):
    """模拟 QwenImage21ZeroCenterRMSNorm（持有 1-D weight 的非 Linear 模块）。"""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))


def test_repair_dequantizes_non_linear_gguf_param():
    src = torch.tensor([1.5, -2.25, 0.5], dtype=torch.bfloat16)
    raw = src.view(torch.uint8).clone()          # (6,) uint8（字节布局）
    p = GGUFParameter(raw, quant_type=GGMLQuantizationType.BF16)
    assert tuple(p.shape) == (6,)
    assert tuple(p.quant_shape) == (3,)

    mod = _Norm(3)
    mod.weight = p

    fixed = _repair_gguf_params(mod, torch.float32)

    assert len(fixed) == 1
    assert tuple(mod.weight.shape) == (3,)
    assert mod.weight.dtype == torch.float32
    assert not isinstance(mod.weight, GGUFParameter)
    assert torch.allclose(mod.weight, src.to(torch.float32), atol=1e-3)


def test_repair_leaves_gguf_linear_untouched():
    """GGUFLinear 的量化权重保持字节态（forward 才反量化），不得被还原。"""
    lin = GGUFLinear(64, 4096, bias=False, compute_dtype=torch.float32)
    raw = torch.zeros((4096, 68), dtype=torch.uint8)  # Q8_0: 64 元素/行 → 68 字节/行
    lin.weight = GGUFParameter(raw, quant_type=GGMLQuantizationType.Q8_0)

    fixed = _repair_gguf_params(lin, torch.float32)
    assert fixed == []
    assert isinstance(lin.weight, GGUFParameter)


def test_repair_nested_module():
    """深层嵌套模块中的非 Linear GGUFParameter 也要被还原。"""
    src = torch.tensor([0.25], dtype=torch.bfloat16)
    raw = src.view(torch.uint8).clone()
    holder = _Norm(1)
    holder.weight = GGUFParameter(raw, quant_type=GGMLQuantizationType.BF16)

    outer = nn.Module()
    outer.inner = holder  # type: ignore[assignment]

    fixed = _repair_gguf_params(outer, torch.float32)
    assert len(fixed) == 1
    assert tuple(holder.weight.shape) == (1,)


def test_repair_noop_on_clean_model():
    m = _Norm(4)  # 普通 float 参数
    assert _repair_gguf_params(m, torch.float32) == []
