"""精度子系统数据模型（分析文档 §11.3）。

对应关系：
    QuantSpec   —— 一档的量化规格（数据，来自 schemes/*.yaml）
    SchemeSpec  —— 一档精度方案（注册表条目）
    LoadPlan    —— resolver + factory 产出的"可执行加载计划"（切换档位时重建）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ===== 后端类型（BackendFactory 分派键）=====
BACKEND_SAFETENSORS = "safetensors"   # T1：from_single_file / from_pretrained（bf16/fp16/fp32）
BACKEND_GGUF = "gguf"                 # T2/T3：GGUFQuantizationConfig（compute=fp32）
BACKEND_INT8 = "int8"                 # T4：torchao / comfy-kitchen

BACKEND_TYPES = (BACKEND_SAFETENSORS, BACKEND_GGUF, BACKEND_INT8)


@dataclass(frozen=True)
class QuantSpec:
    """一档的量化规格（不可变数据）。"""

    backend_type: str               # safetensors | gguf | int8
    dit_dtype: str                  # bf16 | fp16 | q8_0 | q4_k_m | int8
    compute_dtype: str = "auto"     # auto | bf16 | fp16 | fp32（auto 交 DtypeResolver）
    encoder_dtype: str = "fp16"     # bf16 | fp16 | int8 | w4a8
    vae_dtype: str = "fp32"


@dataclass(frozen=True)
class SchemeSpec:
    """一档精度方案（SchemeRegistry 注册条目，来自 schemes/*.yaml）。"""

    id: str                         # t1_bf16 / t2_gguf_q8 / t3_gguf_q4 / t4_int8
    name: str
    description: str
    quantization: QuantSpec
    vram_required: str
    applicable_variants: tuple[str, ...]
    default: bool = False
    steps: int = 25
    guidance: float = 1.0
    available: bool = True           # 依赖是否就绪（如 torchao/comfy-kitchen 未装 → False）
    unavailable_reason: str = ""

    def supports(self, variant: str) -> bool:
        """方案是否适用于指定 variant。"""
        return not self.applicable_variants or variant in self.applicable_variants


@dataclass
class LoadPlan:
    """可执行加载计划（切换档位时重建）。"""

    scheme: SchemeSpec
    variant: str
    device: str                     # 解析后的 "cuda" / "cpu"
    dit_path: Path
    dit_dtype: Any                  # torch.dtype（compute dtype）
    encoder_dtype: Any
    vae_dtype: Any
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "scheme": self.scheme.id,
            "scheme_name": self.scheme.name,
            "variant": self.variant,
            "device": self.device,
            "dit_path": str(self.dit_path),
            "dit_dtype": str(self.dit_dtype),
            "encoder_dtype": str(self.encoder_dtype),
            "vae_dtype": str(self.vae_dtype),
            "notes": self.notes,
        }
