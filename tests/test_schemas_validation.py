"""输入 schema 校验：ImageInputs（宽高 **32 对齐** / 步数 / CFG / 序列长度 / 扩图）。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from engines.qwen_image_21_feeb8130.schemas import ImageInputs


def test_defaults_valid():
    inp = ImageInputs(prompt="a cat")
    assert inp.true_cfg_scale == 1.0
    assert inp.width is None
    assert inp.max_sequence_length == 512


def test_width_height_32_alignment():
    """任务 56：契约由 16 对齐收紧为 **32 对齐**（管线 multiple_of = vae_scale_factor×2 = 32）。

    16 对齐但非 32 的值（1040）必须被拒 —— 否则会被管线静默向下取整
    （1040→1024），出现"页面显示 ≠ 实际输出"。
    """
    ImageInputs(prompt="x", width=1024, height=1024)
    ImageInputs(prompt="x", width=2048, height=2048)
    ImageInputs(prompt="x", width=1056, height=1024)
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", width=1023, height=1024)
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", width=1040, height=1024)   # 16 对齐但非 32


def test_width_height_range():
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", width=240, height=1024)
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", width=2064, height=1024)


def test_steps_range():
    ImageInputs(prompt="x", num_inference_steps=8)
    ImageInputs(prompt="x", num_inference_steps=100)
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", num_inference_steps=0)
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", num_inference_steps=101)


def test_cfg_range():
    ImageInputs(prompt="x", true_cfg_scale=0.0)
    ImageInputs(prompt="x", true_cfg_scale=20.0)
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", true_cfg_scale=-0.1)
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", true_cfg_scale=20.1)


def test_seq_len_range():
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", max_sequence_length=32)
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", max_sequence_length=2048)


def test_scheme_literal():
    ImageInputs(prompt="x", scheme="t2_gguf_q8")
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", scheme="nope")


def test_precision_literal():
    ImageInputs(prompt="x", precision="fp32")
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", precision="float16")


def test_json_schema_exposed():
    """capabilities 里 input_schema 用，必须能生成 JSON schema。"""
    schema = ImageInputs.model_json_schema()
    assert "properties" in schema and "prompt" in schema["properties"]


# ===== 任务 49：三模式扩展字段 =====

def test_mode_literal():
    for m in ("t2i", "i2i", "edit"):
        assert ImageInputs(prompt="x", mode=m).mode == m
    assert ImageInputs(prompt="x").mode is None
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", mode="img2img")


def test_width_height_optional_for_edit_i2i():
    """edit / i2i 允许留空（由管线按条件图推算）。"""
    inp = ImageInputs(prompt="x")
    assert inp.width is None and inp.height is None
    # 只给一边也不报错（交由引擎/管线决定）
    assert ImageInputs(prompt="x", width=1024).height is None


def test_transparent_default_false():
    assert ImageInputs(prompt="x").transparent is False
    assert ImageInputs(prompt="x", transparent=True).transparent is True


def test_denoise_registered_and_bounded():
    """denoise 仅登记（2.1 无 SD 式语义），但有范围校验。"""
    assert ImageInputs(prompt="x", denoise=0.5).denoise == 0.5
    with pytest.raises(ValidationError):
        ImageInputs(prompt="x", denoise=1.5)


def test_images_field_accepts_dicts():
    inp = ImageInputs(prompt="x", images=[{"b64_json": "AAA", "role": "source"}])
    assert inp.images[0]["role"] == "source"
