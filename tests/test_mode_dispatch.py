"""mode 分流与尺寸策略（任务 49 · T2.2 / T2.4）。

用假管线捕获 ``pipe(**kwargs)`` 的调用参数，验证：
    - t2i：无 ``image``，宽高显式
    - edit/i2i：宽高留空 => 不传（由管线按条件图推算）；``image`` 为单张 / list
    - 透明模板注入
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from engines.qwen_image_21_feeb8130.config import EngineConfig
from engines.qwen_image_21_feeb8130.engine import QwenImage21Engine
from engines.qwen_image_21_feeb8130.image_staging import MODE_EDIT, MODE_I2I, MODE_T2I
from engines.qwen_image_21_feeb8130.schemas import ImageInputs


class _Out:
    def __init__(self, img):
        self.images = [img]


class _FakePipe:
    """记录调用参数并返回一张图。"""

    def __init__(self):
        self.kwargs: dict | None = None

    def __call__(self, **kw):
        self.kwargs = kw
        w = kw.get("width") or 512
        h = kw.get("height") or 512
        return _Out(Image.new("RGBA", (w, h), (10, 20, 30, 255)))


@pytest.fixture()
def engine(tmp_path: Path):
    cfg = EngineConfig(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
    )
    eng = QwenImage21Engine(cfg)
    eng._loaded_scheme = "t2_gguf_q8"
    eng.pipeline = _FakePipe()
    # 落盘改为返回伪 URL（不写真文件）
    eng._save_images = lambda images, task_id: [f"/output/image/{task_id}_0.png"]  # type: ignore[method-assign]
    return eng


# ============================ 尺寸策略 ============================

def test_size_default_1024_for_t2i(engine):
    assert engine._effective_size(ImageInputs(prompt="x"), "v", MODE_T2I) == (1024, 1024)


def test_size_explicit_passthrough(engine):
    # 任务 56：宽高契约收紧为 32 对齐（1328 → 1344；原值会被管线静默取整）
    inp = ImageInputs(prompt="x", width=1344, height=928)
    assert engine._effective_size(inp, "v", MODE_EDIT) == (1344, 928)


def test_size_none_for_edit_and_i2i(engine):
    inp = ImageInputs(prompt="x")  # 未指定宽高
    assert engine._effective_size(inp, "v", MODE_EDIT) == (None, None)
    assert engine._effective_size(inp, "v", MODE_I2I) == (None, None)


# ============================ 管线调用参数 ============================

def test_run_pipeline_t2i_no_image(engine):
    inp = ImageInputs(prompt="a cat")
    w, h = engine._effective_size(inp, "v", MODE_T2I)
    engine._run_pipeline(inp, MODE_T2I, [], w, h, 8, "t1", None)
    kw = engine.pipeline.kwargs
    assert "image" not in kw
    assert kw["width"] == 1024 and kw["height"] == 1024
    assert kw["num_inference_steps"] == 8
    assert kw["true_cfg_scale"] == 1.0


def test_run_pipeline_edit_single_image_and_follow_size(engine):
    inp = ImageInputs(prompt="change bg")
    src = Image.new("RGB", (640, 480))
    w, h = engine._effective_size(inp, "v", MODE_EDIT)
    engine._run_pipeline(inp, MODE_EDIT, [src], w, h, 40, "t2", None)
    kw = engine.pipeline.kwargs
    assert not isinstance(kw["image"], list)      # 单张
    assert kw["image"].size == (640, 480)
    assert "width" not in kw and "height" not in kw  # 跟随源图


def test_run_pipeline_edit_with_mask_passes_two(engine):
    inp = ImageInputs(prompt="inpaint")
    src = Image.new("RGB", (640, 480))
    mask = Image.new("L", (640, 480))
    engine._run_pipeline(inp, MODE_EDIT, [src, mask], None, None, 40, "t3", None)
    kw = engine.pipeline.kwargs
    assert isinstance(kw["image"], list) and len(kw["image"]) == 2


def test_run_pipeline_i2i_list(engine):
    inp = ImageInputs(prompt="merge these")
    refs = [Image.new("RGB", (512, 512)) for _ in range(3)]
    engine._run_pipeline(inp, MODE_I2I, refs, None, None, 40, "t4", None)
    kw = engine.pipeline.kwargs
    assert isinstance(kw["image"], list) and len(kw["image"]) == 3


def test_run_pipeline_transparent_template(engine):
    inp = ImageInputs(prompt="a cute dragon sticker", transparent=True)
    engine._run_pipeline(inp, MODE_T2I, [], 1024, 1024, 40, "t5", None)
    kw = engine.pipeline.kwargs
    assert "RGBA image with transparency" in kw["prompt"]
    assert "a cute dragon sticker" in kw["prompt"]


def test_run_pipeline_result_carries_mode(engine):
    inp = ImageInputs(prompt="x")
    res = engine._run_pipeline(inp, MODE_T2I, [], 1024, 1024, 8, "t6", None)
    assert res["mode"] == MODE_T2I
    assert res["width"] == 1024 and res["height"] == 1024
    assert res["image_url"].startswith("/output/image/")


def test_run_pipeline_i2i_reports_actual_size(engine):
    """尺寸跟随条件图时，结果里回填实际产出尺寸。"""
    inp = ImageInputs(prompt="x")
    refs = [Image.new("RGB", (512, 512))] * 2
    res = engine._run_pipeline(inp, MODE_I2I, refs, None, None, 8, "t7", None)
    assert res["width"] == 512 and res["height"] == 512


# ============================ 能力声明 ============================

def test_load_missing_files_keeps_loaded_model(engine, monkeypatch):
    """新档位文件缺失时**不得先卸载已加载模型**（防"READY 但无模型"假就绪态）。

    任务 50 实测踩中：原顺序先 unload 再 check_plan_files，切换失败后引擎停在
    READY 但 pipeline=None → 后续每次生成都报"引擎未加载模型"。
    """
    from engines.qwen_image_21_feeb8130 import engine as eng_mod

    sentinel = engine.pipeline
    unload_calls = {"n": 0}

    async def _fake_unload():
        unload_calls["n"] += 1

    monkeypatch.setattr(engine, "_do_unload", _fake_unload)

    class _Plan:
        scheme = None
        device = "cpu"

    monkeypatch.setattr(engine.schemes, "build_load_plan", lambda **kw: _Plan())

    def _raise(plan, model_dir):
        raise eng_mod.ModelFilesError("模型文件不完整（单测）")

    monkeypatch.setattr(eng_mod, "check_plan_files", _raise)

    import asyncio

    with pytest.raises(eng_mod.ModelFilesError):
        asyncio.run(engine._load_impl("qwen-image-21-official-bf16"))

    assert engine.pipeline is sentinel      # 已加载模型保持不动
    assert unload_calls["n"] == 0           # 绝不能先卸载


def test_save_input_images_writes_served_urls(engine, tmp_path):
    """输入图另存到 output 并回传 /output/... URL（任务中心回显用）。"""
    src = Image.new("RGB", (320, 240), (1, 2, 3))
    mask = Image.new("L", (320, 240), 255)
    items = engine._save_input_images("tid1", [src, mask], ["source", "mask"])
    assert items[0]["url"] == "/output/image/tid1_input_0.png"
    assert items[0]["role"] == "source"
    assert items[1]["role"] == "mask"
    assert (tmp_path / "output" / "image" / "tid1_input_0.png").exists()
    assert (tmp_path / "output" / "image" / "tid1_input_1.png").exists()


def test_capabilities_extra_modes(engine):
    extra = engine.capabilities_extra()
    assert extra["modes"] == ["t2i", "i2i", "edit"]
    assert extra["default_mode"] == "t2i"
    assert extra["max_reference_images"] == 10
    assert extra["supports_transparent"] is True
    assert set(extra["supports_local_edit"]) == {"circle", "paint", "mask"}
