"""本地 variant 扫描 fallback（平台 API 不可达时的降级路径）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from engines.qwen_image_21_feeb8130 import model_scanner
from engines.qwen_image_21_feeb8130 import layout
from engines.qwen_image_21_feeb8130.model_scanner import VARIANT_META, scan_variants_locally


def test_lists_all_variants(tmp_path):
    result = scan_variants_locally(tmp_path)
    assert {r["name"] for r in result} == set(layout.VARIANT_DIT)


def test_fields_aligned_with_platform(tmp_path):
    result = scan_variants_locally(tmp_path)
    for r in result:
        assert set(r) == {"name", "alias", "base_model", "precision", "loader", "vram_required", "available"}
        assert r["loader"] == "diffusers"


def test_unavailable_when_nothing_on_disk(tmp_path):
    result = scan_variants_locally(tmp_path)
    assert all(r["available"] is False for r in result)


def test_available_when_shared_ok_and_dit_present(tmp_path, monkeypatch):
    # 共享组件检查放行（模拟真实场景已下载）
    monkeypatch.setattr(model_scanner, "check_shared_files", lambda _d: None)
    dit_rel = layout.VARIANT_DIT["qwen-image-21-official-gguf-q8"]
    p = tmp_path / dit_rel
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")

    result = {r["name"]: r["available"] for r in scan_variants_locally(tmp_path)}
    assert result["qwen-image-21-official-gguf-q8"] is True
    assert result["qwen-image-21-official-bf16"] is False


def test_shared_missing_marks_all_unavailable(tmp_path, monkeypatch):
    """共享组件不完整 → 全部 variant 不可用（不能因缺共享组件而让 DiT 单独可触发）。"""
    dit_rel = layout.VARIANT_DIT["qwen-image-21-official-gguf-q8"]
    p = tmp_path / dit_rel
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")

    result = scan_variants_locally(tmp_path)
    assert all(r["available"] is False for r in result)


def test_meta_covers_all_variants():
    assert set(VARIANT_META) == set(layout.VARIANT_DIT)


def test_variant_meta_base_models():
    assert VARIANT_META["qwen-image-21-official-gguf-q8"][3] == "Qwen/Qwen-Image-2.1"
    assert "Uncensored" in VARIANT_META["qwen-image-21-uncensored-gguf-q8"][3]
