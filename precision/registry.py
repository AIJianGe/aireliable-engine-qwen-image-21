"""精度档位注册表（SchemeRegistry，分析文档 §11.2 注册表模式）。

从 ``schemes/*.yaml`` 加载 4 档精度方案（T1–T4），集中声明、一处枚举。
新增档位只需新增一个 yaml，引擎主流程零改动（开闭原则）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from .schema import BACKEND_TYPES, QuantSpec, SchemeSpec

logger = logging.getLogger(__name__)


def _dependency_available(backend_type: str, dit_dtype: str) -> tuple[bool, str]:
    """检测档位依赖是否就绪（未装时 UI 标记"不可用"，不阻断其它档）。"""
    if backend_type == "int8":
        try:
            import torchao  # noqa: F401
            return True, ""
        except Exception:  # noqa: BLE001
            pass
        try:
            import comfy_kitchen  # noqa: F401
            return True, ""
        except Exception:  # noqa: BLE001
            return False, "需要 torchao 或 comfy-kitchen（pip install torchao / comfy-kitchen）"
    if dit_dtype == "w4a8":
        try:
            import comfy_kitchen  # noqa: F401
            return True, ""
        except Exception:  # noqa: BLE001
            return False, "需要 comfy-kitchen（W4A8 编码器）"
    return True, ""


class SchemeRegistry:
    """4 档精度方案注册表。"""

    def __init__(self, schemes_dir: Path):
        self._dir = Path(schemes_dir)
        self._cache: dict[str, SchemeSpec] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self._dir.exists():
            logger.warning("schemes 目录不存在：%s", self._dir)
            self._loaded = True
            return
        for yaml_file in sorted(self._dir.glob("scheme_*.yaml")):
            try:
                spec = self._load_yaml(yaml_file)
                if spec is not None:
                    self._cache[spec.id] = spec
                    logger.info(
                        "注册精度档 %s：%s（backend=%s dit=%s compute=%s vram=%s available=%s）",
                        spec.id, spec.name, spec.quantization.backend_type,
                        spec.quantization.dit_dtype, spec.quantization.compute_dtype,
                        spec.vram_required, spec.available,
                    )
            except Exception as e:  # noqa: BLE001
                logger.exception("加载方案文件失败：%s（%s）", yaml_file, e)
        self._loaded = True

    def _load_yaml(self, yaml_path: Path) -> Optional[SchemeSpec]:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not data:
            return None
        q = data.get("quantization", {}) or {}
        backend_type = str(q.get("backend_type", "safetensors"))
        if backend_type not in BACKEND_TYPES:
            raise ValueError(
                f"{yaml_path.name}: 非法 backend_type={backend_type!r}（合法：{BACKEND_TYPES}）"
            )
        dit_dtype = str(q.get("dit_dtype", "fp16"))
        quantization = QuantSpec(
            backend_type=backend_type,
            dit_dtype=dit_dtype,
            compute_dtype=str(q.get("compute_dtype", "auto")),
            encoder_dtype=str(q.get("encoder_dtype", "fp16")),
            vae_dtype=str(q.get("vae_dtype", "fp32")),
        )
        available, reason = _dependency_available(backend_type, dit_dtype)
        inf = data.get("inference", {}) or {}
        return SchemeSpec(
            id=str(data["scheme"]),
            name=str(data.get("name", data["scheme"])),
            description=str(data.get("description", "")),
            quantization=quantization,
            vram_required=str(data.get("vram_required", "—")),
            applicable_variants=tuple(data.get("applicable_variants", []) or []),
            default=bool(data.get("default", False)),
            steps=int(inf.get("steps", 40)),
            guidance=float(inf.get("guidance", 1.0)),
            available=available,
            unavailable_reason=reason,
        )

    def refresh(self) -> None:
        """重新加载（依赖环境变化时）。"""
        self._cache.clear()
        self._loaded = False
        self._ensure_loaded()

    def list(self) -> list[SchemeSpec]:
        """枚举全部档位（按 id 排序）。"""
        self._ensure_loaded()
        return [self._cache[k] for k in sorted(self._cache)]

    def get(self, scheme_id: str) -> Optional[SchemeSpec]:
        """按 id 取档位。"""
        self._ensure_loaded()
        return self._cache.get(scheme_id)

    def default(self) -> Optional[SchemeSpec]:
        """取默认档（yaml 标 default，或 id 排序第一个）。"""
        schemes = self.list()
        for s in schemes:
            if s.default:
                return s
        return schemes[0] if schemes else None
