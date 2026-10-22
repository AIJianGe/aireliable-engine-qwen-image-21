"""档位↔variant 配对回归（任务 51 · R7）。

背景（50 §12）：平台启动引擎未传 `--scheme` → `config.default_scheme` = manifest 默认
`t2_gguf_q8`；而 SDK auto-load 只传 variant、不透传请求档位 → 选 bf16/Q4 这类非-t2 档
variant 时，加载计划拿到"不适用档"直接抛 `ValueError`，导致**每次生成必失败**。

修复：`_load_impl` 在默认档**不适用于**该 variant 时回退到 `auto_select_scheme(variant)`；
显式合法档（如 P7 的 q4 + t3_gguf_q4）必须原样生效。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from engines.qwen_image_21_feeb8130.config import EngineConfig
from engines.qwen_image_21_feeb8130.engine import QwenImage21Engine


class _Spec:
    """最小档位替身：只需 id（`_load_impl` 用 s.id 组 applicable 集合）。"""

    def __init__(self, sid: str):
        self.id = sid


@pytest.fixture()
def engine(tmp_path: Path) -> QwenImage21Engine:
    cfg = EngineConfig(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
    )
    return QwenImage21Engine(cfg)


def _capture_plan(engine, monkeypatch) -> dict:
    """让 build_load_plan 只记录参数后抛错（避免真的去加载权重）。"""
    got: dict = {}

    def _plan(**kw):
        got.update(kw)
        raise RuntimeError("stop-before-real-load")

    monkeypatch.setattr(engine.schemes, "build_load_plan", _plan)
    return got


def test_load_falls_back_to_applicable_scheme(engine, monkeypatch):
    """R7 防呆：默认档不适用于 variant → 回退到适用档（而不是抛"精度档不适用"）。"""
    engine.config.default_scheme = "t2_gguf_q8"          # 平台启动无 --scheme 时的默认档
    monkeypatch.setattr(engine.schemes, "variant_schemes", lambda v: [_Spec("t1_bf16")])
    monkeypatch.setattr(engine.schemes, "auto_select_scheme", lambda v: _Spec("t1_bf16"))
    got = _capture_plan(engine, monkeypatch)

    with pytest.raises(RuntimeError):
        asyncio.run(engine._load_impl("qwen-image-21-official-bf16"))

    assert got["scheme_id"] == "t1_bf16"                 # 已被兜底改写为适用档


def test_load_keeps_explicit_applicable_scheme(engine, monkeypatch):
    """显式合法档（q4 + t3_gguf_q4）必须原样生效——保护 P7 的 T3 复跑。"""
    engine.config.default_scheme = "t3_gguf_q4"
    monkeypatch.setattr(engine.schemes, "variant_schemes", lambda v: [_Spec("t3_gguf_q4")])
    got = _capture_plan(engine, monkeypatch)

    with pytest.raises(RuntimeError):
        asyncio.run(engine._load_impl("qwen-image-21-official-gguf-q4"))

    assert got["scheme_id"] == "t3_gguf_q4"              # 未被兜底改写


def test_load_unknown_variant_keeps_original_error(engine, monkeypatch):
    """variant 无任何适用档（未知 variant）时**不回退**，保留原始报错路径。"""
    engine.config.default_scheme = "t2_gguf_q8"
    monkeypatch.setattr(engine.schemes, "variant_schemes", lambda v: [])
    got = _capture_plan(engine, monkeypatch)

    with pytest.raises(RuntimeError):
        asyncio.run(engine._load_impl("nonexistent-variant"))

    assert got["scheme_id"] == "t2_gguf_q8"              # 原样透传（由 build_load_plan 报错）
