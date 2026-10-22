"""加载后端工厂（BackendFactory，分析文档 §11.2 工厂模式）。

按 ``SchemeSpec.quantization.backend_type`` 产出对应策略实例——
**消灭 ``_load_impl`` 里的 if/else 档位分支**（开闭原则：新增档位只加 spec + backend）。
"""

from __future__ import annotations

from .backends.base import ModelBackend
from .backends.gguf_backend import GgufBackend
from .backends.int8_backend import Int8Backend
from .backends.safetensors_backend import SafetensorsBackend
from .precision.schema import BACKEND_GGUF, BACKEND_INT8, BACKEND_SAFETENSORS, SchemeSpec

_REGISTRY: dict[str, type[ModelBackend]] = {
    BACKEND_SAFETENSORS: SafetensorsBackend,
    BACKEND_GGUF: GgufBackend,
    BACKEND_INT8: Int8Backend,
}


class BackendFactory:
    """后端策略工厂。"""

    @staticmethod
    def create(scheme: SchemeSpec) -> ModelBackend:
        """按量化规格产出后端实例。"""
        backend_type = scheme.quantization.backend_type
        cls = _REGISTRY.get(backend_type)
        if cls is None:
            raise ValueError(
                f"未知 backend_type={backend_type!r}（已注册：{sorted(_REGISTRY)}）"
            )
        return cls(scheme)

    @staticmethod
    def registered() -> list[str]:
        return sorted(_REGISTRY)
