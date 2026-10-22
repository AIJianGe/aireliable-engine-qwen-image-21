"""引擎契约：7 个 _impl 齐备、默认档/默认 variant、采样契约（Turbo 8 步）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from engines.qwen_image_21_feeb8130 import layout
from engines.qwen_image_21_feeb8130.config import EngineConfig
from engines.qwen_image_21_feeb8130.engine import QwenImage21Engine
from engines.qwen_image_21_feeb8130.schemas import ImageInputs

REQUIRED_IMPLS = [
    "_capabilities_impl", "_load_impl", "_unload_impl", "_infer_impl", "_status_impl",
    "_load_lora_impl", "_unload_lora_impl",
]


@pytest.fixture()
def engine(tmp_path: Path):
    cfg = EngineConfig(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
    )
    return QwenImage21Engine(cfg)


def test_identity(engine):
    assert engine.engine_id == "qwen_image_21_feeb8130"
    assert engine.capability == "text-to-image"
    assert engine.engine_type == "image"


def test_all_impls_defined(engine):
    for name in REQUIRED_IMPLS:
        assert callable(getattr(engine, name)), f"缺少 {name}"


def test_default_variant_matches_manifest(engine):
    assert engine.default_variant == "qwen-image-21-official-gguf-q8"


def test_scheme_dicts_four(engine):
    dicts = engine.list_scheme_dicts()
    assert len(dicts) == 4
    ids = {d["id"] for d in dicts}
    assert ids == {"t1_bf16", "t2_gguf_q8", "t3_gguf_q4", "t4_int8"}
    for d in dicts:
        assert "available" in d and "applicable_variants" in d


def test_default_scheme_is_t2(engine):
    assert engine.config.default_scheme == "t2_gguf_q8"
    assert engine.schemes.get_scheme("t2_gguf_q8") is not None


def test_effective_steps_turbo_vs_base(engine):
    inp = ImageInputs(prompt="x")
    assert engine._effective_steps(inp, "qwen-image-21-turbo-gguf-q8") == layout.TURBO_STEPS
    assert engine._effective_steps(inp, "qwen-image-21-official-gguf-q8") == layout.BASE_STEPS
    # 显式覆盖优先
    assert engine._effective_steps(ImageInputs(prompt="x", num_inference_steps=12), "any") == 12


def test_effective_size_default_1024(engine):
    assert engine._effective_size(ImageInputs(prompt="x"), "any") == (1024, 1024)
    # 任务 56：宽高契约 32 对齐（1328 → 1344）
    assert engine._effective_size(ImageInputs(prompt="x", width=1344, height=928), "any") == (1344, 928)


def test_initial_state_clean(engine):
    assert engine.pipeline is None
    assert engine.transformer is None
    assert engine.comps is None
    assert engine._loaded_variant is None


def test_supported_sizes_32_aligned():
    """任务 56：预设尺寸必须 **32 对齐**（管线 multiple_of=32；原 1328x1328 会被静默取整）。"""
    from engines.qwen_image_21_feeb8130.engine import SUPPORTED_SIZES

    for spec in SUPPORTED_SIZES:
        w, h = (int(x) for x in spec.split("x"))
        assert w % 32 == 0 and h % 32 == 0, f"{spec} 非 32 对齐"
        assert 256 <= w <= 2048 and 256 <= h <= 2048, f"{spec} 越界"
    assert "1328x1328" not in SUPPORTED_SIZES, "非 32 对齐的旧预设必须移除"
