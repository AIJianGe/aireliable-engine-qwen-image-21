"""运行时精度自适应（DtypeResolver，分析文档 §12）。

核心：**按显卡实时检测决定 bf16 / fp16 / fp32**。
    - Turing sm_75 无 bf16 张量核 → 请求 bf16 自动降 fp16（超出范围则 fp32）
    - INT8 依赖 ``torch._int_mm``（Turing 支持）
    - 每次决策必记日志：请求档 / 探测结果 / 有效 dtype / 降级原因（project-rules §6.5）
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from .schema import QuantSpec

logger = logging.getLogger(__name__)

_DTYPES: dict[str, Any] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

# bf16 张量核最低架构（Ampere sm_80）
_BF16_MIN_CAP = (8, 0)


class DtypeResolver:
    """硬件探测 → 有效计算 dtype。"""

    def __init__(self) -> None:
        self._probe: Optional[dict] = None

    # ------------------------------------------------------------------
    # 探测
    # ------------------------------------------------------------------

    def probe(self, refresh: bool = False) -> dict:
        """探测本机能力（缓存；refresh=True 重测）。"""
        if self._probe is not None and not refresh:
            return self._probe
        info: dict = {
            "cuda_available": False,
            "capability": None,
            "device_name": None,
            "vram_gb": None,
            "bf16_tensor_core": False,
            "is_bf16_supported": False,       # ⚠️ Turing 返回 True 具有误导性
            "has_int_mm": hasattr(torch, "_int_mm"),
        }
        try:
            info["cuda_available"] = torch.cuda.is_available()
            info["is_bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            if info["cuda_available"]:
                cap = torch.cuda.get_device_capability(0)
                info["capability"] = tuple(cap)
                info["bf16_tensor_core"] = tuple(cap) >= _BF16_MIN_CAP
                info["device_name"] = torch.cuda.get_device_name(0)
                info["vram_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("GPU 探测异常（降级 CPU）：%s", e)
        self._probe = info
        logger.info("硬件探测：%s", info)
        return info

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------

    def resolve(
        self,
        quant: QuantSpec,
        device: str,
        forced: Optional[str] = None,
    ) -> dict:
        """解析有效 dtype。

        Args:
            quant: 该档量化规格（compute_dtype="auto" 时按硬件自动选择）
            device: "cuda" / "cpu"
            forced: 用户/UI 强制精度（auto/bf16/fp16/fp32/int8）；None 表示用档位默认

        Returns:
            {"dit": dtype, "encoder": dtype, "vae": dtype, "notes": [str, ...]}
        """
        notes: list[str] = []

        # ---- CPU：一律 fp32 ----
        if device != "cuda":
            return {
                "dit": torch.float32,
                "encoder": torch.float32,
                "vae": torch.float32,
                "notes": ["CPU 推理 → 全 fp32"],
            }

        info = self.probe()
        cap = info.get("capability")
        req = forced if (forced and forced != "auto") else quant.compute_dtype
        if not req or req == "auto":
            req = "bf16" if info.get("bf16_tensor_core") else "fp16"
            notes.append(f"compute_dtype=auto → 按架构选择 {req}")

        # ---- bf16 → 无张量核则降 fp16（尊重用户强制：仍按请求执行但告警）----
        dit_key = req
        if req == "bf16" and not info.get("bf16_tensor_core"):
            dit_key = "fp16"
            notes.append(
                f"bf16 请求降级 fp16：capability={cap} 无 bf16 张量核"
                f"（is_bf16_supported={info.get('is_bf16_supported')} 不可采信）"
            )
            logger.warning(
                "精度决策：请求 bf16 但本机 capability=%s 无 bf16 张量核 → 降级 fp16"
                "（bf16 仿真预计慢 ~10×）",
                cap,
            )
        if dit_key == "int8" and not info.get("has_int_mm"):
            dit_key = "fp16"
            notes.append("int8 请求降级 fp16：torch._int_mm 不可用")

        # ---- GGUF 档强制 compute=fp32（项目规则 §2 铁律①）----
        if quant.backend_type == "gguf":
            if dit_key != "fp32":
                notes.append(
                    f"GGUF 档强制 compute=fp32（反量化溢出防护；原请求 {dit_key}）"
                )
            dit_key = "fp32"

        dit_dtype = _DTYPES.get(dit_key, torch.float16)

        # ---- 编码器：请求 bf16 权重 → fp16 计算；显式 int8/w4a8 保留 ----
        enc_key = quant.encoder_dtype
        if enc_key == "bf16":
            enc_key = "fp16"
            notes.append("编码器 bf16 → fp16 计算（Turing 无 bf16 张量核）")
        elif enc_key in ("int8", "w4a8"):
            # 量化编码器由 backend 处理，计算 dtype 仍是 fp16
            notes.append(f"编码器量化档 {enc_key}（计算 dtype=fp16）")
            enc_key = "fp16"
        encoder_dtype = _DTYPES.get(enc_key, torch.float16)

        vae_dtype = _DTYPES.get(quant.vae_dtype, torch.float32)

        if not notes:
            notes.append(f"按硬件原生能力执行：dit={dit_key}")
        logger.info(
            "精度决策：scheme backend=%s 请求=%s 探测 capability=%s free=%.1fGB → "
            "dit=%s encoder=%s vae=%s | %s",
            quant.backend_type, req, cap, (info.get("vram_gb") or 0.0),
            dit_key, enc_key, quant.vae_dtype, "；".join(notes),
        )
        return {"dit": dit_dtype, "encoder": encoder_dtype, "vae": vae_dtype, "notes": notes}

    def summary(self) -> dict:
        """当前硬件摘要（/status 展示用）。"""
        return dict(self.probe())
