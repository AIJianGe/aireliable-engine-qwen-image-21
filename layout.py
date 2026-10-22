"""模型文件布局常量（variant × 组件 → 相对 model_dir 的路径）。

单一真相：manifest.yaml 的 ``models.model_files[].target_dir`` 必须与此处一致
（项目规则第一部分 §2：manifest 声明的文件 == load() 实际读取的文件）。
"""

from __future__ import annotations

from pathlib import Path

# 模型根子目录（与 manifest.models.model_layout.base 一致）
MODEL_BASE = "image/qwen-image-2.1/"

# ===== 主 transformer（DiT）—— 每个 variant 一份，互斥 =====
VARIANT_DIT: dict[str, str] = {
    # T1 官方全精度
    "qwen-image-21-official-bf16": "official/qwen-image-2.1-bf16.safetensors",
    # T2 GGUF Q8（★ 主力）
    "qwen-image-21-official-gguf-q8": "transformers/qwen-image-2.1-Q8_0.gguf",
    # T3 GGUF Q4_K_M
    "qwen-image-21-official-gguf-q4": "transformers/qwen-image-2.1-Q4_K_M.gguf",
    # Turbo（8 步契约）
    "qwen-image-21-turbo-gguf-q8": "transformers/qwen-image-2.1-turbo-Q8_0.gguf",
    # 越狱档
    "qwen-image-21-uncensored-gguf-q8": "transformers/qwen-image-2.1-uncensored-Q8_0.gguf",
    # T4 INT8（二期）
    "qwen-image-21-official-int8": "transformers/qwen-image-2.1-int8_convrot.safetensors",
}

# ===== 共享组件（所有 variant 共用一份）=====
TEXT_ENCODER_SUBDIR = "text_encoder"      # Qwen3-VL 8B（bf16 shards + config.json，运行时 cast fp16）
VAE_SUBDIR = "vae"                        # Qwen/Qwen-Image-2.1 vae（config.json + diffusion_pytorch_model.safetensors，RGBA）
TOKENIZER_SUBDIR = "processor"            # Qwen tokenizer/processor（2.1 仓 tokenizer 文件都在 processor/ 下，无 tokenizer/ 目录）
PROCESSOR_SUBDIR = "processor"            # Qwen3VL processor（二期 edit 用）
SCHEDULER_SUBDIR = "scheduler"            # scheduler_config.json

# 生成历史输出子目录（相对 output_root）
OUTPUT_SUBDIR = "image"

# ===== 采样契约（按档位区分；Q0 已实测：QwenImage21Pipeline 默认 40 步 / true_cfg_scale=1.0）=====
TURBO_STEPS = 8
TURBO_GUIDANCE = 1.0
BASE_STEPS = 40
BASE_GUIDANCE = 1.0

# ===== 引擎内置 transformer config（from_single_file 离线加载用）=====
# 背景：from_single_file 不传 config 时会推断 repo 并从 HF 拉 config.json（离线 OSError）。
# 若该目录不存在则省略 config 参数（离线环境会失败 → 需闭环 Q1/Q5 后补入 config.json）。
OFFLINE_CONFIG_DIR = Path(__file__).parent / "configs" / "qwen_transformer"
