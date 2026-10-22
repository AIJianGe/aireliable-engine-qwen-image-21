"""T2/T3 后端：GGUF 单文件（GGUFQuantizationConfig，compute=fp32）。

铁律（项目规则 §2 / 分析文档 §2）：GGUF 反量化在 fp16 下溢出 → NaN → 黑图，
故 compute_dtype **必须 fp32**（DtypeResolver 已对 gguf 档强制 fp32）。
"""

from __future__ import annotations

import logging

from ..layout import OFFLINE_CONFIG_DIR
from ..precision.schema import LoadPlan
from .base import ModelBackend

logger = logging.getLogger(__name__)


def _repair_gguf_params(model, compute_dtype) -> list[str]:
    """还原 diffusers GGUF ``from_single_file`` 未能解码的**非 Linear**参数。

    实测（diffusers 0.41.0 + Qwen-Image-2.1 Q8_0，2026-10-14）：
        ``load_gguf_checkpoint`` 按**字节布局**返回张量（如 BF16[4096] → uint8(8192,)）。
        - nn.Linear 模块会被 ``_replace_with_gguf_linear`` 换成 GGUFLinear，
          forward 时按 quant_type 反量化 → 形状/数值正确；
        - **非 Linear 模块**（如 ``txt_in.text_norm`` 的 QwenImage21ZeroCenterRMSNorm，
          BF16 权重）拿到的是字节张量且原样赋给参数 → 前向
          ``hidden_states(4096) * weight(8192)`` 抛
          RuntimeError: tensor a (4096) must match tensor b (8192)。

    修复：遍历所有**非 GGUFLinear** 模块，凡参数是 GGUFParameter（字节布局）
    则用 ``dequantize_gguf_tensor`` 还原为逻辑形状（``param.quant_shape``），
    再 cast 到计算 dtype。F32 张量本就按逻辑形状载入，不受影响。
    """
    import torch
    from diffusers.quantizers.gguf.utils import (
        GGUFLinear,
        GGUFParameter,
        dequantize_gguf_tensor,
    )

    fixed: list[str] = []
    for mod_name, module in model.named_modules():
        if isinstance(module, GGUFLinear):
            continue  # 量化权重由 GGUFLinear.forward 按需反量化，保持字节态即可
        for pname, param in list(module.named_parameters(recurse=False)):
            if not isinstance(param, GGUFParameter):
                continue
            deq = dequantize_gguf_tensor(param).to(compute_dtype)
            module._parameters[pname] = torch.nn.Parameter(deq, requires_grad=False)
            fixed.append(f"{mod_name}.{pname}: {tuple(param.shape)} → {tuple(deq.shape)}")
            logger.warning(
                "已还原非量化参数：%s.%s: %s → %s（dtype=%s）",
                mod_name, pname, tuple(param.shape), tuple(deq.shape), compute_dtype,
            )
    return fixed


class GgufBackend(ModelBackend):
    """T2/T3：GGUF Q8/Q4 加载策略。"""

    backend_type = "gguf"

    def load_transformer(self, plan: LoadPlan):
        from diffusers import GGUFQuantizationConfig, QwenImage21Transformer2DModel

        kwargs: dict = {
            "quantization_config": GGUFQuantizationConfig(compute_dtype=plan.dit_dtype),
            "torch_dtype": plan.dit_dtype,
        }
        if OFFLINE_CONFIG_DIR.exists():
            kwargs["config"] = str(OFFLINE_CONFIG_DIR)
        else:
            logger.warning(
                "内置 transformer config 目录不存在（%s）——GGUF 离线加载可能失败（Q1）",
                OFFLINE_CONFIG_DIR,
            )
        logger.info(
            "GGUF 加载：from_single_file(%s, compute_dtype=%s)",
            plan.dit_path, plan.dit_dtype,
        )
        model = QwenImage21Transformer2DModel.from_single_file(str(plan.dit_path), **kwargs)
        fixed = _repair_gguf_params(model, plan.dit_dtype)
        if fixed:
            logger.warning("GGUF 参数修复完成（%d 项）：%s", len(fixed), fixed)
        return model
