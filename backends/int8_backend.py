"""T4 后端：INT8（torchao 运行时量化 / comfy-kitchen int8_convrot）。

**二期功能（任务文档 S6，最后做）**。两条"不启动 ComfyUI 服务"的路径：
    4a（先行）：torchao 运行时量化——用未量化权重 + TorchAoConfig
        （项目内 envs/video-ltx25 方案 I8 已在 sm_75 前向验证通过）
    4b（二期）：comfy-kitchen 直接反量化 Comfy-Org 的 int8_convrot 单文件
        （`pip install comfy-kitchen`；需逐层键名映射 + convrot_groupsize 解析）

未装 torchao 时，注册表把该档标记为 available=False，不影响 T1/T2/T3。
"""

from __future__ import annotations

import logging

from ..precision.schema import LoadPlan
from .base import ModelBackend

logger = logging.getLogger(__name__)


class Int8Backend(ModelBackend):
    """T4：INT8 加载策略（二期）。"""

    backend_type = "int8"

    def load_transformer(self, plan: LoadPlan):
        # ---- 4b：Comfy-Org int8_convrot 单文件（comfy-kitchen）----
        if "int8_convrot" in plan.dit_path.name:
            try:
                import comfy_kitchen  # noqa: F401
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    "T4(4b) 需要 comfy-kitchen 反量化 int8_convrot 单文件：\n"
                    f"  pip install comfy-kitchen（或 uv sync --extra int8convrot）\n"
                    f"  本地路径：{plan.dit_path}\n"
                    "详见任务文档 T6.3（Q11）。"
                ) from e
            raise RuntimeError(
                "T4(4b) comfy-kitchen 反量化加载为二期实现（任务文档 T6.3，Q11 未闭环）。"
            )

        # ---- 4a：torchao 运行时量化 ----
        try:
            from diffusers import TorchAoConfig
            from torchao.quantization import Int8DynamicActivationInt8WeightConfig
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "T4(4a) 需要 torchao（运行时 INT8 量化）：\n"
                "  uv sync --extra int8（或 pip install torchao）\n"
                f"  本地路径：{plan.dit_path}\n"
                "详见任务文档 T6.1/T6.2（Q10）。"
            ) from e

        import torch
        from diffusers import QwenImage21Transformer2DModel

        quant_config = TorchAoConfig(Int8DynamicActivationInt8WeightConfig())
        logger.info(
            "T4(4a) torchao INT8 运行时量化：from_single_file(%s)",
            plan.dit_path,
        )
        return QwenImage21Transformer2DModel.from_single_file(
            str(plan.dit_path),
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        )
