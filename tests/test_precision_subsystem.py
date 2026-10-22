"""精度子系统：SchemeRegistry / DtypeResolver / SchemeManager / BackendFactory。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engines.qwen_image_21_feeb8130.factory import BackendFactory
from engines.qwen_image_21_feeb8130.precision.manager import SchemeManager
from engines.qwen_image_21_feeb8130.precision.registry import SchemeRegistry
from engines.qwen_image_21_feeb8130.precision.resolver import DtypeResolver
from engines.qwen_image_21_feeb8130.precision.schema import (
    BACKEND_GGUF,
    BACKEND_INT8,
    BACKEND_SAFETENSORS,
    QuantSpec,
    SchemeSpec,
)

ENGINE_DIR = Path(__file__).resolve().parents[1]
SCHEMES_DIR = ENGINE_DIR / "schemes"

EXPECTED_SCHEMES = {"t1_bf16", "t2_gguf_q8", "t3_gguf_q4", "t4_int8"}


# ============================ Registry ============================

def test_registry_lists_four_schemes():
    reg = SchemeRegistry(SCHEMES_DIR)
    assert {s.id for s in reg.list()} == EXPECTED_SCHEMES


def test_registry_default_is_t2_gguf_q8():
    reg = SchemeRegistry(SCHEMES_DIR)
    assert reg.default().id == "t2_gguf_q8"


def test_registry_backend_types_valid():
    reg = SchemeRegistry(SCHEMES_DIR)
    by_id = {s.id: s for s in reg.list()}
    assert by_id["t1_bf16"].quantization.backend_type == BACKEND_SAFETENSORS
    assert by_id["t2_gguf_q8"].quantization.backend_type == BACKEND_GGUF
    assert by_id["t3_gguf_q4"].quantization.backend_type == BACKEND_GGUF
    assert by_id["t4_int8"].quantization.backend_type == BACKEND_INT8


def test_registry_inference_steps_declared():
    reg = SchemeRegistry(SCHEMES_DIR)
    for s in reg.list():
        assert s.steps >= 1
        assert s.guidance >= 0


def test_int8_scheme_availability_reflects_dependency():
    """T4 未装 torchao/comfy-kitchen → available=False 且有原因（不得虚报可用）。"""
    reg = SchemeRegistry(SCHEMES_DIR)
    s = reg.get("t4_int8")
    try:
        import torchao  # noqa: F401
        has_dep = True
    except Exception:  # noqa: BLE001
        try:
            import comfy_kitchen  # noqa: F401
            has_dep = True
        except Exception:  # noqa: BLE001
            has_dep = False
    assert s.available == has_dep
    if not has_dep:
        assert s.unavailable_reason


def test_registry_refresh_reloads(tmp_path):
    d = tmp_path / "schemes"
    d.mkdir()
    (d / "scheme_x.yaml").write_text(
        "scheme: t9\nname: t9\nquantization:\n  backend_type: gguf\n  dit_dtype: q8_0\n",
        encoding="utf-8",
    )
    reg = SchemeRegistry(d)
    assert [s.id for s in reg.list()] == ["t9"]
    assert reg.refresh() is None
    assert [s.id for s in reg.list()] == ["t9"]


def test_registry_rejects_illegal_backend(tmp_path):
    d = tmp_path / "schemes"
    d.mkdir()
    (d / "scheme_bad.yaml").write_text(
        "scheme: tx\nname: tx\nquantization:\n  backend_type: bogus\n", encoding="utf-8"
    )
    reg = SchemeRegistry(d)
    assert reg.list() == []  # 非法档被记录异常并跳过，不阻断其它档


# ============================ DtypeResolver ============================

def _fake_probe(cap, bf16_tc, has_int_mm=True):
    return {
        "cuda_available": True,
        "capability": cap,
        "device_name": "fake",
        "vram_gb": 48.0,
        "bf16_tensor_core": bf16_tc,
        "is_bf16_supported": True,
        "has_int_mm": has_int_mm,
    }


def test_resolver_cpu_forces_fp32():
    r = DtypeResolver()
    q = QuantSpec(backend_type=BACKEND_SAFETENSORS, dit_dtype="bf16", compute_dtype="auto")
    out = r.resolve(q, "cpu")
    assert out["dit"] == torch.float32
    assert out["encoder"] == torch.float32
    assert out["vae"] == torch.float32


def test_resolver_bf16_degrades_on_turing():
    """Turing sm_75（无 bf16 张量核）请求 bf16 → 降 fp16。"""
    r = DtypeResolver()
    r._probe = _fake_probe((7, 5), False)
    q = QuantSpec(backend_type=BACKEND_SAFETENSORS, dit_dtype="bf16", compute_dtype="auto")
    out = r.resolve(q, "cuda", forced="bf16")
    assert out["dit"] == torch.float16
    assert any("降级" in n or "fp16" in n for n in out["notes"])


def test_resolver_bf16_kept_on_ampere():
    r = DtypeResolver()
    r._probe = _fake_probe((8, 0), True)
    q = QuantSpec(backend_type=BACKEND_SAFETENSORS, dit_dtype="bf16", compute_dtype="auto")
    out = r.resolve(q, "cuda", forced="bf16")
    assert out["dit"] == torch.bfloat16


def test_resolver_auto_picks_fp16_without_bf16_tc():
    r = DtypeResolver()
    r._probe = _fake_probe((7, 5), False)
    q = QuantSpec(backend_type=BACKEND_SAFETENSORS, dit_dtype="bf16", compute_dtype="auto")
    out = r.resolve(q, "cuda")
    assert out["dit"] == torch.float16


def test_resolver_gguf_forces_fp32_ironclad():
    """铁律①：GGUF 档 compute 必须 fp32（防反量化溢出 NaN 黑图）。"""
    r = DtypeResolver()
    r._probe = _fake_probe((8, 0), True)
    q = QuantSpec(backend_type=BACKEND_GGUF, dit_dtype="q8_0", compute_dtype="bf16")
    out = r.resolve(q, "cuda", forced="bf16")
    assert out["dit"] == torch.float32
    assert any("GGUF" in n for n in out["notes"])


def test_resolver_encoder_bf16_becomes_fp16():
    r = DtypeResolver()
    r._probe = _fake_probe((7, 5), False)
    q = QuantSpec(backend_type=BACKEND_SAFETENSORS, dit_dtype="bf16", encoder_dtype="bf16")
    out = r.resolve(q, "cuda")
    assert out["encoder"] == torch.float16


def test_resolver_int8_degrades_without_int_mm():
    r = DtypeResolver()
    r._probe = _fake_probe((7, 5), False, has_int_mm=False)
    q = QuantSpec(backend_type=BACKEND_INT8, dit_dtype="int8", compute_dtype="int8")
    out = r.resolve(q, "cuda", forced="int8")
    assert out["dit"] == torch.float16


# ============================ SchemeManager ============================

def test_manager_build_plan_t2_default():
    m = SchemeManager(SCHEMES_DIR)
    model_dir = Path("models/image/qwen-image-2.1")
    plan = m.build_load_plan("qwen-image-21-official-gguf-q8", model_dir, device="cpu")
    assert plan.scheme.id == "t2_gguf_q8"
    assert plan.device == "cpu"
    assert plan.dit_dtype == torch.float32
    assert plan.dit_path.name == "qwen-image-2.1-Q8_0.gguf"
    parts = plan.dit_path.parts
    assert parts[-4:] == ("image", "qwen-image-2.1", "transformers", "qwen-image-2.1-Q8_0.gguf")


def test_manager_rejects_unknown_variant():
    m = SchemeManager(SCHEMES_DIR)
    with pytest.raises(ValueError):
        m.build_load_plan("not-a-variant", Path("models/image/qwen-image-2.1"))


def test_manager_rejects_inapplicable_scheme():
    """T1 只适用 bf16 档，套到 gguf-q8 variant 上必须拒绝。"""
    m = SchemeManager(SCHEMES_DIR)
    with pytest.raises(ValueError):
        m.build_load_plan(
            "qwen-image-21-official-gguf-q8",
            Path("models/image/qwen-image-2.1"),
            scheme_id="t1_bf16",
        )


def test_manager_variant_schemes_mapping():
    m = SchemeManager(SCHEMES_DIR)
    ids = {s.id for s in m.variant_schemes("qwen-image-21-official-gguf-q8")}
    assert ids == {"t2_gguf_q8"}
    ids_bf16 = {s.id for s in m.variant_schemes("qwen-image-21-official-bf16")}
    assert ids_bf16 == {"t1_bf16"}


def test_manager_plan_as_dict_serializable():
    m = SchemeManager(SCHEMES_DIR)
    plan = m.build_load_plan("qwen-image-21-official-gguf-q8", Path("models/x"), device="cpu")
    d = plan.as_dict()
    assert d["scheme"] == "t2_gguf_q8"
    assert isinstance(d["dit_path"], str)


# ============================ BackendFactory ============================

def test_factory_maps_backend_types():
    m = SchemeManager(SCHEMES_DIR)
    by_id = {s.id: s for s in m.list_schemes()}
    assert type(BackendFactory.create(by_id["t1_bf16"])).__name__ == "SafetensorsBackend"
    assert type(BackendFactory.create(by_id["t2_gguf_q8"])).__name__ == "GgufBackend"
    assert type(BackendFactory.create(by_id["t4_int8"])).__name__ == "Int8Backend"


def test_factory_rejects_unknown_backend():
    bogus = SchemeSpec(
        id="tx", name="tx", description="",
        quantization=QuantSpec(backend_type="bogus", dit_dtype="x"),
        vram_required="—", applicable_variants=(),
    )
    with pytest.raises(ValueError):
        BackendFactory.create(bogus)


def test_factory_registered_list():
    assert set(BackendFactory.registered()) == {BACKEND_SAFETENSORS, BACKEND_GGUF, BACKEND_INT8}
