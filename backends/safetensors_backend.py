"""T1 后端：官方 safetensors 全精度（bf16/fp16/fp32）。

做法：QwenImage21Transformer2DModel.from_single_file(..., torch_dtype=<resolver 决定>)
    - 本机 sm_75 无 bf16 张量核 → DtypeResolver 已把 bf16 降为 fp16（溢出则 fp32）
"""

from __future__ import annotations

import logging

from ..layout import OFFLINE_CONFIG_DIR
from ..precision.schema import LoadPlan
from .base import ModelBackend

logger = logging.getLogger(__name__)


class SafetensorsBackend(ModelBackend):
    """T1：官方全精度权重加载策略。"""

    backend_type = "safetensors"

    def load_transformer(self, plan: LoadPlan):
        from diffusers import QwenImage21Transformer2DModel

        kwargs: dict = {"torch_dtype": plan.dit_dtype}
        if OFFLINE_CONFIG_DIR.exists():
            kwargs["config"] = str(OFFLINE_CONFIG_DIR)
        else:
            logger.warning(
                "内置 transformer config 目录不存在（%s）——离线加载可能失败，"
                "需闭环 Q1/Q5 后补入 config.json",
                OFFLINE_CONFIG_DIR,
            )
        logger.info(
            "T1 加载：from_single_file(%s, torch_dtype=%s)",
            plan.dit_path, plan.dit_dtype,
        )
        return QwenImage21Transformer2DModel.from_single_file(str(plan.dit_path), **kwargs)
