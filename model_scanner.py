"""variant 本地扫描 fallback（平台 API 不可达时的降级路径）。

原则（项目规则第三部分 §3 / 分析文档 §5.9）：
    - 本地 fallback 只检查**共享组件标志文件 + DiT 文件存在性**，
      用于让 variant 在 UI 中**可见**（可触发下载），不比平台 API 更严格。
    - 返回结构对齐平台 VariantInfo 字段。
"""

from __future__ import annotations

import logging
from pathlib import Path

from .layout import VARIANT_DIT
from .pipeline_builder import ModelFilesError, check_shared_files

logger = logging.getLogger(__name__)

# variant → (显示别名, 精度, 显存需求, base_model)
VARIANT_META: dict[str, tuple[str, str, str, str]] = {
    "qwen-image-21-official-bf16": ("Qwen-Image2.1-FP", "bf16", "34GB", "Qwen/Qwen-Image-2.1"),
    "qwen-image-21-official-gguf-q8": ("Qwen-Image2.1-Q8", "q8_0", "20GB", "Qwen/Qwen-Image-2.1"),
    "qwen-image-21-official-gguf-q4": ("Qwen-Image2.1-Q4", "q4_k_m", "12GB", "Qwen/Qwen-Image-2.1"),
    "qwen-image-21-turbo-gguf-q8": ("Qwen-Image2.1-Turbo", "q8_0", "20GB", "Qwen/Qwen-Image-2.1-Turbo"),
    "qwen-image-21-uncensored-gguf-q8": ("Qwen-Image2.1-UC", "q8_0", "20GB", "KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF"),
    "qwen-image-21-official-int8": ("Qwen-Image2.1-INT8", "int8", "18GB", "Comfy-Org/Qwen-Image-2.1"),
}


def scan_variants_locally(model_dir: Path) -> list[dict]:
    """本地扫描全部 variant（exists + size 检查；供 /capabilities fallback）。

    available = 共享组件齐全 且 该 variant 的 DiT 文件在盘。
    """
    shared_ok = True
    try:
        check_shared_files(model_dir)
    except ModelFilesError as e:
        shared_ok = False
        logger.warning("共享组件不完整（variant 均标记不可用）：%s", str(e).splitlines()[0])

    result: list[dict] = []
    for variant, rel in VARIANT_DIT.items():
        alias, precision, vram, base_model = VARIANT_META.get(
            variant, (variant, "", "—", "")
        )
        dit_path = model_dir / rel
        dit_ok = dit_path.exists() and dit_path.stat().st_size > 0
        result.append({
            "name": variant,
            "alias": alias,
            "base_model": base_model,
            "precision": precision,
            "loader": "diffusers",
            "vram_required": vram,
            "available": bool(shared_ok and dit_ok),
        })
    logger.info(
        "本地扫描完成（model_dir=%s）：shared=%s %s",
        model_dir, shared_ok, {r["name"]: r["available"] for r in result},
    )
    return result
