"""输入图 staging + 模式解析（任务 49 · T1.4 / T2.4）。"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from engines.qwen_image_21_feeb8130 import image_staging as stg
from engines.qwen_image_21_feeb8130.image_staging import (
    MAX_REFERENCE_IMAGES,
    MODE_EDIT,
    MODE_I2I,
    MODE_T2I,
    ImageStagingError,
    describe_roles,
    merge_mask_overlay,
    normalize_mode,
    resolve_mode,
    split_edit_inputs,
    stage_images,
    validate_mode,
)


# ============================ 模式解析 ============================

def test_normalize_mode_aliases():
    assert normalize_mode("t2i") == MODE_T2I
    assert normalize_mode("T2I") == MODE_T2I
    assert normalize_mode("text_to_image") == MODE_T2I
    assert normalize_mode("image_generation") == MODE_T2I
    assert normalize_mode("img2img") == MODE_I2I
    assert normalize_mode("image_to_image") == MODE_I2I
    assert normalize_mode("image_editing") == MODE_EDIT
    assert normalize_mode("image_edit") == MODE_EDIT
    assert normalize_mode("nonsense") is None
    assert normalize_mode(None) is None


def test_resolve_mode_infer_from_image_count():
    assert resolve_mode(None, None, 0) == MODE_T2I
    assert resolve_mode(None, None, 1) == MODE_EDIT
    assert resolve_mode(None, None, 2) == MODE_I2I
    assert resolve_mode(None, None, 5) == MODE_I2I


def test_resolve_mode_explicit_wins_over_workflow():
    assert resolve_mode("i2i", "edit", 1) == MODE_I2I
    assert resolve_mode(None, "edit", 3) == MODE_EDIT


def test_resolve_mode_unknown_falls_back_to_inference():
    assert resolve_mode("bogus", "bogus", 2) == MODE_I2I


# ============================ 模式校验 ============================

def test_validate_t2i_rejects_images():
    with pytest.raises(ImageStagingError) as ei:
        validate_mode(MODE_T2I, [object()], ["source"])
    assert "不接受输入图" in str(ei.value)


def test_validate_edit_needs_at_least_one():
    validate_mode(MODE_EDIT, [object()], ["source"])  # 1 张 OK
    with pytest.raises(ImageStagingError):
        validate_mode(MODE_EDIT, [], [])


def test_validate_edit_allows_two_with_mask():
    validate_mode(MODE_EDIT, [object(), object()], ["source", "mask"])


def test_validate_i2i_needs_two():
    with pytest.raises(ImageStagingError) as ei:
        validate_mode(MODE_I2I, [object()], ["reference"])
    assert "需要 2" in str(ei.value)
    validate_mode(MODE_I2I, [object(), object()], ["reference", "reference"])


def test_validate_rejects_over_hard_cap():
    n = MAX_REFERENCE_IMAGES + 1
    with pytest.raises(ImageStagingError) as ei:
        validate_mode(MODE_I2I, [object()] * n, ["reference"] * n)
    assert "超上限" in str(ei.value)


def test_validate_unknown_mode():
    with pytest.raises(ImageStagingError):
        validate_mode("nope", [], [])


def test_describe_roles():
    assert describe_roles(["source", "mask", None]) == {"source": 1, "mask": 1, "unspecified": 1}


# ============================ staging ============================

def _png_bytes(mode="RGBA", size=(512, 512), color=(255, 0, 0, 128)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_stage_b64_preserves_rgba(tmp_path):
    payload = [{"b64_json": base64.b64encode(_png_bytes()).decode(), "role": "source"}]
    imgs, roles = stage_images(payload, tmp_path)
    assert len(imgs) == 1
    assert imgs[0].mode == "RGBA"          # 透明输入不丢 alpha
    assert roles == ["source"]


def test_stage_data_url_prefix(tmp_path):
    b64 = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
    imgs, _ = stage_images([{"b64_json": b64}], tmp_path)
    assert imgs[0].size == (512, 512)


def test_stage_local_path(tmp_path):
    p = tmp_path / "src.png"
    p.write_bytes(_png_bytes())
    imgs, roles = stage_images([{"path": str(p), "role": "reference"}], tmp_path)
    assert imgs[0].size == (512, 512)
    assert roles == ["reference"]


def test_stage_url_local_path(tmp_path):
    """任务 60：url 字段塞本地绝对路径 → 按本地路径读取（绕过网关直连引擎兜底）。"""
    p = tmp_path / "src.png"
    p.write_bytes(_png_bytes())
    imgs, _ = stage_images([{"url": str(p), "role": "source"}], tmp_path)
    assert imgs[0].size == (512, 512)


def test_stage_path_out_of_root_rejected(tmp_path):
    """任务 60：path 越出允许根 → ImageStagingError（防任意文件读取）。"""
    outside = tmp_path.parent / "outside_secret.png"
    outside.write_bytes(_png_bytes())
    with pytest.raises(ImageStagingError) as ei:
        stage_images([{"path": str(outside)}], tmp_path)
    assert "越界" in str(ei.value)


def test_stage_url_local_out_of_root_rejected(tmp_path):
    """任务 60：url 塞越界本地路径 → ImageStagingError。"""
    outside = tmp_path.parent / "outside_secret2.png"
    outside.write_bytes(_png_bytes())
    with pytest.raises(ImageStagingError) as ei:
        stage_images([{"url": str(outside)}], tmp_path)
    assert "越界" in str(ei.value)


def test_stage_missing_source_raises(tmp_path):
    with pytest.raises(ImageStagingError) as ei:
        stage_images([{"role": "source"}], tmp_path)
    assert "缺少 url/b64_json/path" in str(ei.value)


def test_stage_bad_base64_raises(tmp_path):
    with pytest.raises(ImageStagingError):
        stage_images([{"b64_json": "!!!not-base64!!!"}], tmp_path)


def test_stage_too_small_raises(tmp_path):
    small = base64.b64encode(_png_bytes(size=(64, 64))).decode()
    with pytest.raises(ImageStagingError) as ei:
        stage_images([{"b64_json": small}], tmp_path)
    assert "过小" in str(ei.value)


def test_stage_undecodable_raises(tmp_path):
    junk = base64.b64encode(b"not an image at all").decode()
    with pytest.raises(ImageStagingError):
        stage_images([{"b64_json": junk}], tmp_path)


def test_stage_keeps_order(tmp_path):
    payload = [{"b64_json": base64.b64encode(_png_bytes(color=(i * 40, 0, 0, 255))).decode()} for i in range(3)]
    imgs, roles = stage_images(payload, tmp_path)
    assert len(imgs) == 3 and len(roles) == 3


# ============================ 独立 mask 合并（任务 50 冒烟修复） ============================

def test_merge_mask_overlay_tints_region():
    src = Image.new("RGB", (200, 200), (0, 0, 255))       # 纯蓝源图
    mask = Image.new("L", (200, 200), 0)
    for x in range(50, 150):
        for y in range(50, 150):
            mask.putpixel((x, y), 255)
    out = merge_mask_overlay(src, mask)
    assert out.mode == "RGB"
    # 区域中心被叠红（红通道上升、蓝通道下降）
    r, g, b = out.getpixel((100, 100))
    assert r > 60 and b < 200
    # 区域外保持纯蓝
    assert out.getpixel((10, 10)) == (0, 0, 255)


def test_split_edit_inputs_merges_mask_and_drops_mask():
    src = Image.new("RGB", (200, 200), (0, 0, 255))
    mask = Image.new("L", (200, 200), 255)
    kept, roles = split_edit_inputs([src, mask], ["source", "mask"])
    assert len(kept) == 1 and roles == ["source"]


def test_split_edit_inputs_noop_without_mask():
    a, b = Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))
    kept, roles = split_edit_inputs([a, b], ["reference", "reference"])
    assert kept == [a, b] and roles == ["reference", "reference"]
