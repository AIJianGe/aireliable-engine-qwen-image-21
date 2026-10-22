"""加载后端策略族（分析文档 §11.2 策略模式）。

    ModelBackend（抽象基类）
        ├── SafetensorsBackend   T1：官方 bf16/fp16/fp32 权重（from_single_file）
        ├── GgufBackend          T2/T3：GGUF 单文件（GGUFQuantizationConfig，compute=fp32）
        └── Int8Backend          T4：torchao 运行时量化 / comfy-kitchen int8_convrot（二期）

差异只存在于"如何加载 DiT"，共享组件与管线组装由 pipeline_builder 统一处理。
"""

from .base import ModelBackend

__all__ = ["ModelBackend"]
