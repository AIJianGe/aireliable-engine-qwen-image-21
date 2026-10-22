"""embeds dtype 桥接回归测试（本轮修复：bool 掩码被误 cast 成浮点 → torch.where 崩）。"""

from __future__ import annotations

import torch

from engines.qwen_image_21_feeb8130.pipeline_builder import (
    _wrap_encoder_embeds_dtype,
    _wrap_vae_encode_dtype,
)


class _FakePipe:
    """模拟 QwenImage21Pipeline 的 prompt 编码入口。"""

    def __init__(self):
        self._execution_device = torch.device("cpu")
        self.captured_device = None

    def _get_qwen_prompt_embeds(self, prompt=None, image=None, device=None):
        self.captured_device = device
        embeds = torch.randn(1, 7, 4096, dtype=torch.float16)
        attn_mask = torch.ones(1, 7, dtype=torch.long)
        img_mask = torch.zeros(1, 7, dtype=torch.bool)
        return embeds, attn_mask, img_mask


def test_bridge_casts_only_floating_tensors():
    pipe = _FakePipe()
    _wrap_encoder_embeds_dtype(pipe, "cpu", torch.float32)

    embeds, attn_mask, img_mask = pipe._get_qwen_prompt_embeds(prompt=["x"])

    # 浮点 embeds → 目标 dtype
    assert embeds.dtype == torch.float32
    # bool 掩码必须保持 bool（否则 transformer 的 torch.where 抛错）
    assert img_mask.dtype == torch.bool
    # 整型注意力掩码不被改成 float
    assert attn_mask.dtype == torch.long


def test_bridge_redirects_device_arg():
    pipe = _FakePipe()
    _wrap_encoder_embeds_dtype(pipe, "cpu", torch.float32)
    pipe._get_qwen_prompt_embeds(prompt=["x"], device="cuda:0")
    assert str(pipe.captured_device) == "cpu"


def test_bridge_accepts_single_return():
    class _SinglePipe:
        _execution_device = torch.device("cpu")

        def encode_prompt(self, prompt=None):
            return torch.randn(2, dtype=torch.float16)

    pipe = _SinglePipe()
    _wrap_encoder_embeds_dtype(pipe, "cpu", torch.float32)
    out = pipe.encode_prompt(prompt=["x"])
    assert out.dtype == torch.float32


class _FakeVaePipe:
    """模拟 QwenImage21Pipeline 的 VAE 编码入口（fp32 VAE + fp16 条件图）。"""

    def __init__(self, vae_dtype=torch.float32):
        self.vae = torch.nn.Conv2d(3, 3, kernel_size=1).to(vae_dtype)
        self.received_dtype = None
        self.received_device = None

    def _encode_vae_image(self, image, generator):
        self.received_dtype = image.dtype
        self.received_device = image.device
        # 模拟 diffusers：返回的 latents 与输入同 dtype（此处即 vae.dtype）
        return torch.randn(1, 16, 8, 8, dtype=image.dtype)


def test_vae_bridge_casts_image_to_vae_dtype_and_back():
    """T1 fp16 档：条件图必须升到 vae.dtype(fp32)，latents 还原 dit dtype(fp16)。"""
    pipe = _FakeVaePipe(vae_dtype=torch.float32)
    _wrap_vae_encode_dtype(pipe, torch.float16)

    out = pipe._encode_vae_image(torch.randn(1, 3, 64, 64, dtype=torch.float16), None)

    assert pipe.received_dtype == torch.float32      # 输入按 VAE dtype（避免 conv2d 冲突）
    assert out.dtype == torch.float16                # 输出还原 dit dtype（供与噪声 cat）


def test_vae_bridge_keeps_fp32_when_dit_is_fp32():
    """T2/T3 GGUF 档：dit=fp32 与 vae=fp32 一致，桥接不改变 dtype。"""
    pipe = _FakeVaePipe(vae_dtype=torch.float32)
    _wrap_vae_encode_dtype(pipe, torch.float32)

    out = pipe._encode_vae_image(torch.randn(1, 3, 64, 64, dtype=torch.float16), None)

    assert pipe.received_dtype == torch.float32
    assert out.dtype == torch.float32
