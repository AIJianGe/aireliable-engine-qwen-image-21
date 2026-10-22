"""加载后端抽象基类（策略模式，分析文档 §11.2）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..precision.schema import LoadPlan, SchemeSpec


class ModelBackend(ABC):
    """DiT 加载策略基类。

    子类只需实现 ``load_transformer``；共享组件加载与管线组装由
    pipeline_builder 统一处理，避免各档重复代码。
    """

    backend_type: str = ""

    def __init__(self, scheme: SchemeSpec):
        self.scheme = scheme

    @abstractmethod
    def load_transformer(self, plan: LoadPlan) -> Any:
        """加载 DiT transformer（返回 diffusers 模型实例）。"""

    def describe(self) -> dict:
        return {
            "backend_type": self.backend_type,
            "scheme": self.scheme.id,
            "dit_dtype": self.scheme.quantization.dit_dtype,
            "compute_dtype": self.scheme.quantization.compute_dtype,
        }
