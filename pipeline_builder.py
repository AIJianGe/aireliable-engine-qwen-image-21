"""Qwen-Image-2.1 管线组件组装器。

职责（engine load() 的底层实现）：
    1. 文件完整性校验（exists + size ≥ 阈值；缺失抛 ModelFilesError 指引下载，
       禁止在 load 路径做任何网络下载——项目规则第二部分 §2）
    2. 共享组件加载：Qwen3-VL-8B 编码器 / AutoencoderKLQwenImage21 VAE /
       tokenizer / scheduler（全档共享，整个引擎生命周期只加载一份）
    3. DiT transformer 加载由 precision 层的 Backend（策略）完成：
       T1 SafetensorsBackend / T2+T3 GgufBackend / T4 Int8Backend

复用既有 7007 引擎（qwen_image_5e8f3a21/pipeline_builder.py）已验证的三条铁律：
    ① GGUF compute 必须 fp32（反量化溢出 → NaN → 黑图）
    ② 文本编码器 bf16 权重 → fp16 计算
    ③ 编码器 embeds 必须桥接到执行 dtype（_wrap_vl_embeds_dtype）
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .layout import (
    SCHEDULER_SUBDIR,
    TEXT_ENCODER_SUBDIR,
    TOKENIZER_SUBDIR,
    VAE_SUBDIR,
)
from .precision.schema import LoadPlan

logger = logging.getLogger(__name__)

# ===== 最小大小阈值（字节，防残文件；约为实际大小的 90%）=====
MIN_SIZES: dict[str, int] = {
    "model-00001-of-00004.safetensors": int(4.4 * 1024**3),
    "model-00002-of-00004.safetensors": int(4.4 * 1024**3),
    "model-00003-of-00004.safetensors": int(4.4 * 1024**3),
    "model-00004-of-00004.safetensors": int(2.3 * 1024**3),
    "diffusion_pytorch_model.safetensors": int(600 * 1024**2),
}

# 共享组件文件清单（相对 model_dir 的路径 → MIN_SIZES 键；空串 = 存在性检查即可）
SHARED_FILES: list[tuple[str, str]] = [
    (f"{TEXT_ENCODER_SUBDIR}/config.json", ""),
    (f"{TEXT_ENCODER_SUBDIR}/model.safetensors.index.json", ""),
    (f"{TEXT_ENCODER_SUBDIR}/model-00001-of-00004.safetensors", "model-00001-of-00004.safetensors"),
    (f"{TEXT_ENCODER_SUBDIR}/model-00002-of-00004.safetensors", "model-00002-of-00004.safetensors"),
    (f"{TEXT_ENCODER_SUBDIR}/model-00003-of-00004.safetensors", "model-00003-of-00004.safetensors"),
    (f"{TEXT_ENCODER_SUBDIR}/model-00004-of-00004.safetensors", "model-00004-of-00004.safetensors"),
    (f"{VAE_SUBDIR}/config.json", ""),
    (f"{VAE_SUBDIR}/diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.safetensors"),
    (f"{TOKENIZER_SUBDIR}/vocab.json", ""),
    (f"{TOKENIZER_SUBDIR}/merges.txt", ""),
    (f"{TOKENIZER_SUBDIR}/tokenizer_config.json", ""),
    (f"{TOKENIZER_SUBDIR}/special_tokens_map.json", ""),
    (f"{TOKENIZER_SUBDIR}/tokenizer.json", ""),
    (f"{TOKENIZER_SUBDIR}/added_tokens.json", ""),
    (f"{TOKENIZER_SUBDIR}/preprocessor_config.json", ""),
    (f"{TOKENIZER_SUBDIR}/video_preprocessor_config.json", ""),
    (f"{TOKENIZER_SUBDIR}/chat_template.jinja", ""),
    (f"{SCHEDULER_SUBDIR}/scheduler_config.json", ""),
]

# 编码器 fp16 显存需求估算（Qwen3-VL-8B bf16 ~17.5GB → fp16 同量级）
ENCODER_VRAM_NEED_GB = 18.0


class ModelFilesError(RuntimeError):
    """模型文件缺失/不完整（load() 校验失败抛出，信息指引模型管理界面下载）。"""


# ============================================================
# 文件完整性校验（项目规则：exists + size ≥ 阈值，禁止 load 内下载）
# ============================================================

def check_shared_files(model_dir: Path) -> None:
    """校验共享组件文件完整性（缺失抛 ModelFilesError）。"""
    missing: list[str] = []
    for rel, min_key in SHARED_FILES:
        path = model_dir / rel
        min_size = MIN_SIZES.get(min_key, 1)
        if not path.exists():
            missing.append(rel)
        elif min_size > 1 and path.stat().st_size < min_size:
            missing.append(f"{rel}（大小不足：{path.stat().st_size / 1024**3:.2f}GB）")
    if missing:
        raise ModelFilesError(_format_missing(missing, model_dir, "共享组件（Qwen3-VL 编码器 / VAE / tokenizer / scheduler）"))


def check_plan_files(plan: LoadPlan, model_dir: Path) -> None:
    """校验 DiT 文件（plan.dit_path）存在且大小合理。"""
    path = plan.dit_path
    if not path.exists():
        raise ModelFilesError(_format_missing(
            [str(path.relative_to(model_dir)) if path.is_relative_to(model_dir) else str(path)],
            model_dir, f"DiT transformer（{plan.scheme.id} / {plan.variant}）",
        ))
    # GGUF 至少数 GB；safetensors 单文件同量级（阈值放宽到 1GB 防误判）
    if path.suffix in (".gguf", ".safetensors") and path.stat().st_size < 1 * 1024**3:
        raise ModelFilesError(_format_missing(
            [f"{path.name}（大小不足：{path.stat().st_size / 1024**3:.2f}GB）"],
            model_dir, f"DiT transformer（{plan.scheme.id}）",
        ))


def _format_missing(missing: list[str], model_dir: Path, group: str) -> str:
    """构造带下载指引的缺失错误信息。"""
    return (
        f"模型文件不完整（{group}），缺失 {len(missing)} 个文件：\n"
        f"  {missing}\n\n"
        f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮，选择镜像源后下载。\n"
        f"本地路径：{model_dir}\n"
        f"仓库 ID：Qwen/Qwen-Image-2.1（官方）/ unsloth/Qwen-Image-2.1-GGUF（GGUF）/ "
        f"Comfy-Org/Qwen-Image-2.1（int8_convrot）"
    )


# ============================================================
# 共享组件加载（单份常驻）
# ============================================================

@dataclass
class SharedComponents:
    """已加载的共享组件集合（整个引擎生命周期一份）。"""

    vae: Any = None
    text_encoder: Any = None
    tokenizer: Any = None
    processor: Any = None          # Qwen3VL Processor（QwenImage21Pipeline 需要 processor 而非 tokenizer）
    scheduler_config: dict = field(default_factory=dict)
    encoder_device: str = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self.vae is not None and self.text_encoder is not None


def decide_encoder_device(device: str, vram_need_gb: float = ENCODER_VRAM_NEED_GB) -> str:
    """决策编码器放置（默认 GPU，剩余显存不足则降 CPU 内存驻留）。"""
    if not device.startswith("cuda"):
        return "cpu"
    try:
        import torch
        free, _total = torch.cuda.mem_get_info()
        free_gb = free / 1024**3
    except Exception:  # noqa: BLE001
        return "cpu"
    if free_gb >= vram_need_gb + 2.0:
        logger.info("编码器放置决策：GPU（剩余 %.1fGB ≥ %.1fGB）", free_gb, vram_need_gb + 2.0)
        return device
    logger.warning(
        "编码器放置决策：CPU 内存驻留（剩余显存 %.1fGB < %.1fGB，编码每次 +几秒）",
        free_gb, vram_need_gb + 2.0,
    )
    return "cpu"


def load_shared_components(model_dir: Path, device: str, plan: LoadPlan) -> SharedComponents:
    """加载全部共享组件（load() 时调用一次，之后常驻）。"""
    check_shared_files(model_dir)

    comps = SharedComponents()

    t0 = time.perf_counter()
    comps.vae = _load_vae(model_dir / VAE_SUBDIR, device, plan.vae_dtype)
    logger.info("VAE 加载完成（%.1fs，%s，dtype=%s）", time.perf_counter() - t0, device, plan.vae_dtype)

    comps.encoder_device = decide_encoder_device(device)
    t0 = time.perf_counter()
    comps.text_encoder = _load_text_encoder(model_dir / TEXT_ENCODER_SUBDIR, comps.encoder_device, plan.encoder_dtype)
    logger.info(
        "Qwen3-VL-8B 编码器加载完成（%.1fs，放置=%s，dtype=%s）",
        time.perf_counter() - t0, comps.encoder_device, plan.encoder_dtype,
    )

    comps.tokenizer = _load_tokenizer(model_dir / TOKENIZER_SUBDIR)
    comps.processor = _load_processor(model_dir / TOKENIZER_SUBDIR)
    cfg_path = model_dir / SCHEDULER_SUBDIR / "scheduler_config.json"
    comps.scheduler_config = json.loads(cfg_path.read_text(encoding="utf-8"))
    logger.info("tokenizer / scheduler 配置加载完成")
    return comps


def _load_vae(vae_dir: Path, device: str, dtype: Any):
    """VAE → AutoencoderKLQwenImage21（diffusers 目录格式 from_pretrained）。"""
    from diffusers import AutoencoderKLQwenImage21
    vae = AutoencoderKLQwenImage21.from_pretrained(str(vae_dir), torch_dtype=dtype)
    return vae.to(device).eval()


def _load_text_encoder(te_dir: Path, enc_device: str, dtype: Any):
    """Qwen3-VL-8B 编码器（bf16 权重 → 计算 dtype）。

    本机实证（7007 同类）：bf16 权重 cast fp16 编码干净（nan=0 / inf=0）。
    """
    from transformers import AutoModelForMultimodalLM
    try:
        te = AutoModelForMultimodalLM.from_pretrained(str(te_dir), torch_dtype=dtype)
    except Exception as e:  # noqa: BLE001
        logger.warning("AutoModelForMultimodalLM 加载失败（%s），回退 Qwen3VLForConditionalGeneration", e)
        from transformers import Qwen3VLForConditionalGeneration
        te = Qwen3VLForConditionalGeneration.from_pretrained(str(te_dir), torch_dtype=dtype)
    return te.to(enc_device).eval()


def _load_tokenizer(tok_dir: Path):
    """Qwen tokenizer（AutoTokenizer 自适应 Qwen2/Qwen3 词表）。"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(tok_dir))


def _load_processor(proc_dir: Path):
    """Qwen3VL Processor（AutoProcessor：tokenizer + image processor）。

    QwenImage21Pipeline.__init__ 签名要求 processor（scheduler/vae/text_encoder/
    processor/transformer/sample_sigmas），故必须加载 processor 而非 tokenizer。
    """
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str(proc_dir))


def make_scheduler(scheduler_config: dict):
    """用 scheduler_config 构造 FlowMatchEulerDiscreteScheduler。"""
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)


# ============================================================
# 管线组装
# ============================================================
# 注意：组装后【不调用 pipe.to(device)】——组件异构放置（编码器可能驻 CPU），
# 各组件在加载时已就位。

def _wrap_encoder_embeds_dtype(pipe: Any, encoder_device: str, out_dtype: Any) -> None:
    """包装管线 prompt 编码方法：编码输出统一 cast 到执行 dtype。

    黑图/崩溃修复配套（7007 同类实证）：
    1. latents dtype 跟随 prompt_embeds.dtype：编码器 fp16，GGUF forward_native
       不做输入 dtype cast（本环境无 kernels ggml CUDA ops）→ 输入与权重 dtype
       不一致 RuntimeError → embeds cast fp32 使 latents 与 transformer 输入一致
    2. 编码器驻 CPU 的降级场景：把编码输入重定向到编码器所在设备
    """
    import inspect

    target_name = None
    for cand in ("_get_qwen_prompt_embeds", "encode_prompt"):
        if hasattr(pipe, cand):
            target_name = cand
            break
    if target_name is None:
        logger.warning("管线未找到 prompt 编码方法，embeds dtype 桥接跳过（若出黑图请检查）")
        return

    orig = getattr(pipe, target_name)
    enc_dev = __import__("torch").device(encoder_device)

    def wrapped(*args, **kwargs):
        try:
            bound = inspect.signature(orig).bind(*args, **kwargs)
            if "device" in bound.arguments:
                bound.arguments["device"] = enc_dev
            bound.arguments.pop("dtype", None)
            out = orig(*bound.args, **bound.kwargs)
        except TypeError:
            out = orig(*args, **kwargs)
        # 单返回值（embeds）或多返回值（embeds, mask）
        # ⚠️ 只 cast 浮点张量：`_get_qwen_prompt_embeds` 同时返回 bool 掩码
        # （encoder_attention_mask / image_pad_mask），若一并 cast 成 fp32，
        # transformer 前向 `torch.where(img_mask, ...)` 会抛
        # "where expected condition to be a boolean tensor, but got Float"。
        exec_dev = getattr(pipe, "_execution_device", None)

        def _bridge(t):
            if not hasattr(t, "to"):
                return t
            if t.is_floating_point():
                t = t.to(out_dtype)
            return t.to(exec_dev) if exec_dev is not None else t

        if isinstance(out, tuple):
            return tuple(_bridge(t) for t in out)
        return _bridge(out)

    setattr(pipe, target_name, wrapped)
    logger.info(
        "embeds dtype 桥接已启用（%s：编码器=%s → embeds=%s）",
        target_name, encoder_device, out_dtype,
    )


def _wrap_vae_encode_dtype(pipe: Any, out_dtype: Any) -> None:
    """VAE 编码 dtype 桥接（T1 bf16→fp16 条件图崩溃修复）。

    diffusers ``QwenImage21Pipeline.prepare_latents`` 对条件图统一执行
    ``image.to(device, dtype=prompt_embeds.dtype)``，而本引擎 VAE 按 **fp32** 加载
    （数值稳定，见 t1 scheme ``vae_dtype: fp32``）。

    - T2/T3（GGUF）档 dit 被强制 fp32 → 条件图 cast fp32 → 与 fp32 VAE 一致 → 正常；
    - **T1 档 dit=fp16** → 条件图 cast fp16 → 喂给 fp32 VAE 的 conv → 抛
      ``RuntimeError: Input type (struct c10::Half) and bias type (float) should be the same``
      （i2i/edit 必崩，t2i 无条件图不受影响）。

    解码路径 diffusers 已自带 ``latents.to(self.vae.dtype)``（故 t2i 出图正常）；
    仅编码路径缺此转换，在此补齐：输入按 ``vae.dtype``、输出还原 ``dit`` dtype
    （供后续与 fp16 噪声 latents ``cat``）。
    """
    import torch as _torch

    orig = pipe._encode_vae_image

    def wrapped(image, generator):
        ref = next(pipe.vae.parameters())
        image = image.to(device=ref.device, dtype=ref.dtype)
        latents = orig(image, generator)
        if isinstance(latents, _torch.Tensor) and latents.is_floating_point():
            latents = latents.to(out_dtype)
        return latents

    setattr(pipe, "_encode_vae_image", wrapped)
    logger.info(
        "VAE 编码 dtype 桥接已启用（条件图 → vae.dtype，latents → dit=%s）", out_dtype,
    )


def build_pipeline(comps: SharedComponents, transformer: Any, plan: LoadPlan):
    """组装文生图管线（QwenImage21Pipeline + 已加载 transformer）。"""
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline(
        scheduler=make_scheduler(comps.scheduler_config),
        vae=comps.vae,
        text_encoder=comps.text_encoder,
        processor=comps.processor,
        transformer=transformer,
    )
    # Windows 子进程 stderr 非控制台句柄，tqdm 写入会抛 OSError；进度走 SSE
    pipe.set_progress_bar_config(disable=True)
    _wrap_encoder_embeds_dtype(pipe, comps.encoder_device, plan.dit_dtype)
    _wrap_vae_encode_dtype(pipe, plan.dit_dtype)
    return pipe
