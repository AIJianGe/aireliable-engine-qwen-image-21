"""扩图（outpaint）与尺寸契约单测（任务 56）。

覆盖：
    1. `expanded_size()` —— 各方向画布尺寸（含源图非 32 对齐时的向上补齐）
    2. `expand_canvas()` —— 源图像素保留、贴合位置、边缘色填充、RGBA 处理
    3. 非法入参（方向/扩幅）
    4. `ExpandSpec` schema 校验（方向 literal / 扩幅范围与对齐）
    5. `QwenImage21Engine._validate_expand()` —— 模式/蒙版/张数/画布上限
    6. `_effective_size()` —— 显式尺寸优先于默认（i2i/edit 留空才跟随条件图）

为什么补这些断言（本轮实测教训）：
    - 仅给更大的输出尺寸 ≠ 扩图（模型会整幅重绘，原图区差异 69.8/255）；
      必须"扩展画布当输入"（原图区差异 6.7/255、接缝差 0.046）
    - 管线 `multiple_of = vae_scale_factor × 2 = 32`，画布必须 32 对齐，否则输出被静默取整
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from engines.qwen_image_21_feeb8130.config import EngineConfig
from engines.qwen_image_21_feeb8130.engine import QwenImage21Engine
from engines.qwen_image_21_feeb8130.image_staging import (
    EXPAND_DIRECTIONS,
    MODE_EDIT,
    align_up,
    expand_canvas,
    expanded_size,
)
from engines.qwen_image_21_feeb8130.schemas import (
    SIZE_ALIGN,
    SIZE_MAX,
    ExpandSpec,
    ImageInputs,
)

SRC_W, SRC_H = 1024, 1024


@pytest.fixture()
def engine(tmp_path: Path):
    cfg = EngineConfig(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
    )
    return QwenImage21Engine(cfg)


def _make_src(w: int = SRC_W, h: int = SRC_H, mode: str = "RGB") -> Image.Image:
    """构造可判别的测试源图：左半黑、右半白（便于验证"原图像素被保留在正确位置"）。"""
    channels = 4 if mode == "RGBA" else 3
    arr = np.zeros((h, w, channels), dtype=np.uint8)
    arr[:, w // 2:] = 255
    if channels == 4:
        arr[:, :, 3] = 255
    return Image.fromarray(arr, mode)


# ===== 1. 画布尺寸 =====

def test_align_up_multiple_of_32():
    assert SIZE_ALIGN == 32
    assert align_up(1024) == 1024
    assert align_up(1000) == 1024
    assert align_up(1025) == 1056


@pytest.mark.parametrize(
    ("direction", "amount", "expected"),
    [
        ("right", 320, (1344, 1024)),
        ("left", 320, (1344, 1024)),
        ("up", 320, (1024, 1344)),
        ("down", 320, (1024, 1344)),
        ("all", 128, (1280, 1280)),
    ],
)
def test_expanded_size_by_direction(direction, amount, expected):
    assert expanded_size(SRC_W, SRC_H, direction, amount) == expected


def test_expanded_size_aligns_unaligned_source():
    """源图非 32 对齐时，画布先向上补齐再扩（避免管线静默取整）。"""
    assert expanded_size(1000, 1000, "right", 320) == (1344, 1024)   # 1024+320
    assert expanded_size(1000, 1000, "all", 128) == (1280, 1280)     # 1024+256


def test_expanded_size_rejects_bad_input():
    with pytest.raises(ValueError):
        expanded_size(SRC_W, SRC_H, "diagonal", 320)
    with pytest.raises(ValueError):
        expanded_size(SRC_W, SRC_H, "right", 0)


# ===== 2. 画布几何 / 像素保留 =====

def test_expand_canvas_right_keeps_source_pixels_at_left():
    src = _make_src()
    out = expand_canvas(src, "right", 320)
    assert out.size == (1344, 1024)
    # 源图原样落在左侧（黑左半 → x=100 仍为黑；白右半 → x=900 仍为白）
    assert out.getpixel((100, 500)) == src.getpixel((100, 500)) == (0, 0, 0)
    assert out.getpixel((900, 500)) == src.getpixel((900, 500)) == (255, 255, 255)
    # 新增区（x>1024）为右边缘均值色 → 源图右边缘是白 → 255
    assert out.getpixel((1300, 500)) == (255, 255, 255)


def test_expand_canvas_left_places_source_at_right():
    src = _make_src()
    out = expand_canvas(src, "left", 320)
    assert out.size == (1344, 1024)
    # 源图贴右侧：源图 (0,500)=黑 → 落在 x=320 处
    assert out.getpixel((320 + 100, 500)) == (0, 0, 0)
    # 左侧新增区用左边缘均值色（黑）填充
    assert out.getpixel((100, 500)) == (0, 0, 0)


def test_expand_canvas_up_places_source_at_bottom():
    src = _make_src()
    out = expand_canvas(src, "up", 256)
    assert out.size == (1024, 1280)
    assert out.getpixel((500, 256 + 100)) == src.getpixel((500, 100))
    # 上侧新增区 = 上边缘均值色（该行左半黑右半白 → 均值 ≈127）
    top = out.getpixel((5, 5))[0]
    assert 120 <= top <= 135, f"上侧填充色异常：{top}"


def test_expand_canvas_all_centers_source_and_32_aligned():
    src = _make_src()
    out = expand_canvas(src, "all", 128)
    assert out.size == (1280, 1280)
    assert out.size[0] % SIZE_ALIGN == 0 and out.size[1] % SIZE_ALIGN == 0
    # 源图居中：偏移 (128,128)
    assert out.getpixel((128 + 100, 500 + 0)) == (0, 0, 0)


def test_expand_canvas_handles_unaligned_source():
    """非 32 对齐源图：画布 32 对齐 + 源图像素仍完整保留（只在其右/下侧补最多 31px）。"""
    src = _make_src(1000, 900)
    out = expand_canvas(src, "right", 320)
    assert out.size == (1344, 928)             # align_up(1000)+320, align_up(900)
    assert out.getpixel((0, 0)) == src.getpixel((0, 0))
    assert out.getpixel((999, 899)) == src.getpixel((999, 899))
    assert out.size[0] % SIZE_ALIGN == 0 and out.size[1] % SIZE_ALIGN == 0


def test_expand_canvas_rgba_keeps_mode_and_opaque_fill():
    src = _make_src(mode="RGBA")
    out = expand_canvas(src, "right", 256)
    assert out.mode == "RGBA"
    assert out.getpixel((1200, 500))[3] == 255      # 新增区不透明
    assert out.getpixel((300, 500))[3] == 255


# ===== 3/4. schema 校验 =====

def test_expand_spec_defaults_and_literal():
    spec = ExpandSpec()
    assert spec.direction == "right" and spec.amount == 256 and spec.auto_prompt is True
    assert ExpandSpec(direction="all", amount=128).direction == "all"
    with pytest.raises(Exception):
        ExpandSpec(direction="diagonal")


def test_expand_spec_amount_bounds_and_alignment():
    with pytest.raises(Exception):
        ExpandSpec(amount=16)        # < 32
    with pytest.raises(Exception):
        ExpandSpec(amount=2048)      # > 1024
    with pytest.raises(Exception):
        ExpandSpec(amount=48)        # 非 32 对齐（48 = 1.5×32）
    assert ExpandSpec(amount=1024).amount == 1024


def test_image_inputs_expand_field_optional():
    assert ImageInputs(prompt="x").expand is None
    inp = ImageInputs(prompt="x", expand={"direction": "right", "amount": 320})
    assert inp.expand is not None and inp.expand.direction == "right"


def test_width_height_must_be_32_aligned():
    """管线 multiple_of=32 → 16 对齐但非 32 的值（如 1040）必须被拒（否则静默取整）。"""
    ImageInputs(prompt="x", width=1024, height=1024)
    ImageInputs(prompt="x", width=1056, height=1056)
    with pytest.raises(Exception):
        ImageInputs(prompt="x", width=1040, height=1024)   # 1040 = 32.5×32
    with pytest.raises(Exception):
        ImageInputs(prompt="x", width=1023, height=1024)


# ===== 5. 引擎侧扩图校验 =====

def test_validate_expand_requires_edit_mode(engine):
    exp = ExpandSpec(direction="right", amount=320)
    src = [_make_src()]
    assert engine._validate_expand("t2i", src, ["unspecified"], exp) is not None
    assert engine._validate_expand("i2i", src, ["reference"], exp) is not None
    assert engine._validate_expand(MODE_EDIT, src, ["source"], exp) is None


def test_validate_expand_rejects_mask_and_multi_image(engine):
    exp = ExpandSpec(direction="right", amount=320)
    err_mask = engine._validate_expand(MODE_EDIT, [_make_src()], ["source", "mask"], exp)
    assert err_mask and "蒙版" in err_mask
    err_multi = engine._validate_expand(MODE_EDIT, [_make_src(), _make_src()], ["source", "source"], exp)
    assert err_multi and "1 张源图" in err_multi
    err_zero = engine._validate_expand(MODE_EDIT, [], [], exp)
    assert err_zero is not None


def test_validate_expand_rejects_canvas_over_max(engine):
    exp = ExpandSpec(direction="all", amount=1024)
    err = engine._validate_expand(MODE_EDIT, [_make_src(1024, 1024)], ["source"], exp)
    assert err and str(SIZE_MAX) in err        # 1024 + 2×1024 = 3072 > 2048 → 必拒


def test_validate_expand_boundary_ok(engine):
    """right 扩 1024：画布正好 2048（上限边界）应通过。"""
    exp = ExpandSpec(direction="right", amount=1024)
    assert engine._validate_expand(MODE_EDIT, [_make_src(1024, 1024)], ["source"], exp) is None


# ===== 6. 尺寸解析 =====

def test_effective_size_explicit_wins(engine):
    for mode in ("t2i", "i2i", "edit"):
        w, h = engine._effective_size(ImageInputs(prompt="x", width=1344, height=768), "v", mode)
        assert (w, h) == (1344, 768)


def test_effective_size_defaults(engine):
    assert engine._effective_size(ImageInputs(prompt="x"), "v", "t2i") == (1024, 1024)
    assert engine._effective_size(ImageInputs(prompt="x"), "v", "edit") == (None, None)
    assert engine._effective_size(ImageInputs(prompt="x"), "v", "i2i") == (None, None)


def test_expand_directions_constant_consistent():
    assert set(EXPAND_DIRECTIONS) == {"left", "right", "up", "down", "all"}
