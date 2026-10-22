"""精度档位管理器（SchemeManager，分析文档 §11.4）。

职责：
    - 枚举 4 档（list_schemes）
    - 按 variant 校验适用性（validate_scheme_for_variant）
    - 组装可执行加载计划（build_load_plan：注册表取 spec → DtypeResolver 定 dtype → 路径解析）
    - 切换守卫（ensure_unloaded_before_switch 由 engine 调用）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..layout import MODEL_BASE, VARIANT_DIT
from .registry import SchemeRegistry
from .resolver import DtypeResolver
from .schema import LoadPlan, SchemeSpec

logger = logging.getLogger(__name__)


class SchemeManager:
    """4 档精度方案管理器。"""

    def __init__(self, schemes_dir: Path, resolver: Optional[DtypeResolver] = None):
        self.registry = SchemeRegistry(schemes_dir)
        self.resolver = resolver or DtypeResolver()

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def list_schemes(self) -> list[SchemeSpec]:
        """枚举全部档位。"""
        return self.registry.list()

    def get_scheme(self, scheme_id: str) -> Optional[SchemeSpec]:
        return self.registry.get(scheme_id)

    def default_scheme(self) -> Optional[SchemeSpec]:
        return self.registry.default()

    def auto_select_scheme(self, variant: str) -> Optional[SchemeSpec]:
        """按 variant 选默认档（优先 yaml default，否则第一个适用档）。"""
        d = self.registry.default()
        if d is not None and d.supports(variant):
            return d
        for s in self.registry.list():
            if s.supports(variant):
                return s
        return d

    def validate_scheme_for_variant(self, scheme: SchemeSpec, variant: str) -> bool:
        """校验档位是否适用于该 variant。"""
        ok = scheme.supports(variant)
        if not ok:
            logger.warning(
                "档位 %s 不适用于 variant %s（适用列表：%s）",
                scheme.id, variant, list(scheme.applicable_variants),
            )
        return ok

    def variant_schemes(self, variant: str) -> list[SchemeSpec]:
        """该 variant 可用的全部档位。"""
        return [s for s in self.registry.list() if s.supports(variant)]

    # ------------------------------------------------------------------
    # 加载计划
    # ------------------------------------------------------------------

    def build_load_plan(
        self,
        variant: str,
        model_dir: Path,
        scheme_id: Optional[str] = None,
        precision: Optional[str] = None,
        device: str = "auto",
    ) -> LoadPlan:
        """组装可执行加载计划。

        Raises:
            ValueError: variant 非法 / 档位非法 / 档位不适用
        """
        if variant not in VARIANT_DIT:
            raise ValueError(
                f"未知 variant：{variant!r}（可用：{sorted(VARIANT_DIT)}）"
            )

        scheme = self.get_scheme(scheme_id) if scheme_id else None
        if scheme_id and scheme is None:
            raise ValueError(
                f"未知精度档：{scheme_id!r}"
                f"（可用：{[s.id for s in self.registry.list()]}）"
            )
        if scheme is None:
            scheme = self.auto_select_scheme(variant)
        if scheme is None:
            raise ValueError("未找到任何精度档（schemes/ 为空？）")
        if not self.validate_scheme_for_variant(scheme, variant):
            raise ValueError(
                f"精度档 {scheme.id} 不适用于 variant {variant}"
                f"（适用：{list(scheme.applicable_variants)}）"
            )

        resolved_device = self._resolve_device(device)
        dtypes = self.resolver.resolve(
            scheme.quantization, resolved_device, forced=precision,
        )

        dit_path = (model_dir / VARIANT_DIT[variant]).resolve()
        plan = LoadPlan(
            scheme=scheme,
            variant=variant,
            device=resolved_device,
            dit_path=dit_path,
            dit_dtype=dtypes["dit"],
            encoder_dtype=dtypes["encoder"],
            vae_dtype=dtypes["vae"],
            notes=list(dtypes["notes"]),
        )
        logger.info("加载计划：%s", plan.as_dict())
        return plan

    # ------------------------------------------------------------------

    def _resolve_device(self, device: str) -> str:
        if device in ("cpu",):
            return "cpu"
        if device and device.startswith("cuda"):
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def hardware_summary(self) -> dict:
        """硬件摘要（/status 用）。"""
        return self.resolver.summary()

    @staticmethod
    def model_base() -> str:
        return MODEL_BASE
