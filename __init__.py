"""Qwen-Image-2.1 图像引擎（engine_id: qwen_image_21_feeb8130，integrated / CustomEngine 全自定义范式）。

任务编号 48，见 ANALYSIS-TASKS/48-QwenImage2.1图像生成引擎/。
支持 4 档精度模式（T1 官方 BF16 / T2 GGUF Q8 ★ / T3 GGUF Q4 / T4 INT8），
运行时按显卡自适应 dtype。
"""

__all__ = ["__version__"]

__version__ = "1.0.2"
