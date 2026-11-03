"""Qwen-Image-2.1 图像引擎推理核心（CustomEngine / integrated 全自定义范式）。

继承 ``CustomEngine``，实现 7 个 ``_xxx_impl``。**只编排，不关心量化细节**：
    - precision/（SchemeRegistry + DtypeResolver + SchemeManager）→ 4 档精度方案
    - factory.py（BackendFactory）+ backends/（策略族）→ DiT 加载
    - pipeline_builder.py → 共享组件加载 + 管线组装 + embeds 桥接

8 端点契约（server.py 注册路由）：
    /capabilities  /load  /unload  /infer(SSE)  /status  /lora/load  /lora/unload
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from aireliable_engine import (
    CapabilitiesResponse,
    CustomEngine,
    EngineLoadError,
    InferRequest,
    LoadResponse,
    LoraLoadResponse,
    LoraUnloadResponse,
    SSEStream,
    StatusResponse,
    UnloadResponse,
    VariantInfo,
    make_error_frame,
    make_final_frame,
    make_frame,
)

from .config import EngineConfig
from .factory import BackendFactory
from .image_staging import (
    LOCAL_EDIT_METHODS,
    MAX_REFERENCE_IMAGES,
    MODE_EDIT,
    MODE_T2I,
    SOFT_REFERENCE_IMAGES,
    SUPPORTED_MODES,
    ImageStagingError,
    describe_roles,
    expand_canvas,
    expanded_size,
    resolve_mode,
    split_edit_inputs,
    stage_images,
    validate_mode,
)
from .layout import OUTPUT_SUBDIR, VARIANT_DIT
from .model_scanner import scan_variants_locally
from .pipeline_builder import (
    ModelFilesError,
    SharedComponents,
    build_pipeline,
    check_plan_files,
    load_shared_components,
)
from .precision.manager import SchemeManager
from .precision.schema import LoadPlan
from .schemas import (
    EXPAND_DIRECTION_LABELS,
    EXPAND_PROMPT_TEMPLATE,
    SIZE_MAX,
    TRANSPARENT_PROMPT_TEMPLATE,
    ExpandSpec,
    ImageInputs,
)
from .task_reporting import QwenImage21TaskReporting, fetch_history_page

logger = logging.getLogger(__name__)

# capabilities 声明的常用输出尺寸（含 2K 原生）
# ⚠️ 必须 **32 对齐**（任务 56 源码核实）：管线 `QwenImage21Pipeline.__call__` 内
#    `multiple_of = vae_scale_factor * 2 = 32`，非 32 对齐的宽高会被**静默向下取整**
#    （如 1360→1344、1328→1312）→ "页面显示 ≠ 实际输出"。原表的 `1328x1328`（41.5×32）
#    正属此类，已替换为电商常用的 4:3 / 3:4（1024x768 / 768x1024，均 32 对齐）。
#    顺序即 UI 预设按钮顺序（webui/app.js renderRatios 取 slice(0,7)）。
SUPPORTED_SIZES: list[str] = [
    "1024x1024",                                   # 1:1（默认）
    "1024x768", "768x1024",                        # 4:3 / 3:4（电商主图 / 竖版详情页）
    "1664x928", "928x1664",                        # 16:9 / 9:16（官方 1M px 档）
    "1664x704", "704x1664",                        # 21:9 / 9:21（官方）
    "2048x2048",                                   # 1:1 2K（原生上限）
    "1152x896", "896x1152",                        # 9:7 / 7:9（官方）
]


class QwenImage21Engine(CustomEngine):
    """Qwen-Image-2.1 图像引擎（t2i，4 档精度 T1–T4）。"""

    def __init__(self, config: EngineConfig):
        super().__init__(
            engine_id=config.engine_id,
            engine_name=config.engine_name,
            capability=config.capability,
            engine_type=config.engine_type,
        )
        self.config = config
        self.output_root: Path = config.output_root

        # manifest 默认 variant
        self.default_variant: Optional[str] = self._read_default_variant_from_manifest(
            Path(__file__).parent / "manifest.yaml"
        )

        # ===== 精度子系统（注册表 + 解析器 + 管理器）=====
        self.schemes = SchemeManager(config.schemes_dir)

        # ===== 运行时组件 =====
        self.comps: Optional[SharedComponents] = None
        self.transformer: Any = None
        self.pipeline: Any = None
        self.plan: Optional[LoadPlan] = None
        self._loaded_variant: Optional[str] = None
        self._loaded_scheme: Optional[str] = None
        self._device: Optional[str] = None
        self._load_time: Optional[float] = None
        self._last_used_at: Optional[float] = None

        self.task_reporting = QwenImage21TaskReporting(config.engine_id)

        logger.info(
            "QwenImage21Engine 初始化：config=%s default_variant=%s schemes=%s",
            config.to_dict(), self.default_variant,
            [s.id for s in self.schemes.list_schemes()],
        )

    # ============================================================
    # /capabilities
    # ============================================================

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        variants: list[VariantInfo] = []
        platform_variants = await self.get_variants()
        if platform_variants is not None:
            import os
            logger.info(
                "variant 扫描路径：平台 API（AIRELIABLE_CORE_URL=%s，%d 个 variant）",
                os.environ.get("AIRELIABLE_CORE_URL"), len(platform_variants),
            )
            for v in platform_variants:
                variants.append(VariantInfo(
                    name=v.get("name", ""),
                    base_model=v.get("base_model", "Qwen/Qwen-Image-2.1"),
                    precision=v.get("precision", ""),
                    loader=v.get("loader", "diffusers"),
                    vram_required=v.get("vram_required", "20GB"),
                    available=bool(v.get("available", False)),
                ))
        else:
            logger.warning("variant 扫描路径：本地 fallback（平台 API 不可达或未注入 AIRELIABLE_CORE_URL）")
            variants = [VariantInfo(**r) for r in scan_variants_locally(self.config.model_dir)]

        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=variants,
            loras=[],
            supported_sizes=SUPPORTED_SIZES,
            default_variant=self.default_variant,
            input_schema=ImageInputs.model_json_schema(),
        )

    @staticmethod
    def capabilities_extra() -> dict:
        """三模式能力声明（任务 49）。

        SDK 的 ``CapabilitiesResponse`` 无 modes 字段，故由 ``server.py`` 的
        ``/capabilities`` 处理器注入（与 variants dump 同一手法）。
        """
        return {
            "modes": list(SUPPORTED_MODES),
            "default_mode": MODE_T2I,
            "max_reference_images": MAX_REFERENCE_IMAGES,
            "soft_reference_images": SOFT_REFERENCE_IMAGES,
            "supports_transparent": True,
            "supports_local_edit": list(LOCAL_EDIT_METHODS),
        }

    # ============================================================
    # /load
    # ============================================================

    async def _load_impl(self, variant: Optional[str]) -> LoadResponse:
        scheme_id = self.config.default_scheme
        variant_name = variant or self.default_variant
        if not variant_name:
            raise ModelFilesError("manifest 未声明 default_variant，且未显式指定 variant")

        logger.info("load() 开始：variant=%s scheme=%s", variant_name, scheme_id)

        # R7 防呆（任务 51）：启动默认档（无 --scheme 时 = manifest 默认 t2_gguf_q8）可能
        # **不适用于**本次请求的 variant（如 bf16 / q4）。此时回退到该 variant 的适用档，
        # 而不是直接抛「精度档不适用」——否则平台启动的引擎只要选非-t2 档 variant 就必然
        # 失败（SDK auto-load 只传 variant，不透传请求档位，见 50 §12 E3/E4）。
        # ⚠️ 仅在"不适用"时回退；显式合法档（如 P7 的 q4 + t3_gguf_q4）原样生效。
        applicable = {s.id for s in self.schemes.variant_schemes(variant_name)}
        if scheme_id and applicable and scheme_id not in applicable:
            auto = self.schemes.auto_select_scheme(variant_name)
            if auto is not None:
                logger.warning(
                    "档位 %s 不适用于 variant %s（适用：%s）→ 回退到适用档 %s（R7 防呆）",
                    scheme_id, variant_name, sorted(applicable), auto.id,
                )
                scheme_id = auto.id

        plan = self.schemes.build_load_plan(
            variant=variant_name,
            model_dir=self.config.model_dir,
            scheme_id=scheme_id,
            precision=self.config.precision,
            device=self.config.device,
        )

        # 「只校验不下载」的文件完整性检查（项目规则 §2）——**必须放在 unload 之前**。
        # 根因（任务 50 实测踩中）：原顺序是先 unload 旧模型、再在 _blocking_load 里
        # check_plan_files；新档位文件缺失时旧模型已被卸掉 → 引擎停在「READY 但无模型」
        # 的假就绪态（SDK 只在 IDLE 时 auto-load）→ 之后每次生成都报
        # "引擎未加载模型（load 未完成或已卸载）"，用户观感"点生成没反应"。
        # 先校验：不通过则直接抛错，当前已加载模型保持不动。
        await asyncio.to_thread(check_plan_files, plan, self.config.model_dir)

        # 切换守卫：确认新档位文件齐全后，才 unload 旧模型（防跨档残留）
        if self.pipeline is not None or self.transformer is not None:
            logger.warning("load()：检测到残留状态，先 unload 再加载（档位切换守卫）")
            await self._do_unload()

        t0 = time.perf_counter()

        def _blocking_load() -> SharedComponents:
            backend = BackendFactory.create(plan.scheme)
            logger.info("加载后端：%s（%s）", backend.describe(), plan.scheme.id)
            comps = load_shared_components(self.config.model_dir, plan.device, plan)
            transformer = backend.load_transformer(plan)
            transformer = transformer.to(plan.device).eval()
            self.transformer = transformer
            self.pipeline = build_pipeline(comps, transformer, plan)
            return comps

        self.comps = await asyncio.to_thread(_blocking_load)
        self.plan = plan
        self._device = plan.device
        self._loaded_variant = variant_name
        self._loaded_scheme = plan.scheme.id
        self._load_time = time.perf_counter() - t0

        logger.info(
            "load() 完成：variant=%s scheme=%s device=%s dit_dtype=%s encoder=%s 耗时=%.1fs 显存=%.1fGB",
            variant_name, plan.scheme.id, plan.device, plan.dit_dtype,
            plan.encoder_dtype, self._load_time, self._vram_used_mb() / 1024,
        )
        return LoadResponse(
            variant=variant_name,
            state="ready",
            vram_used=self._vram_used_str(),
            load_time=round(self._load_time, 1),
        )

    # ============================================================
    # /unload
    # ============================================================

    async def _unload_impl(self) -> UnloadResponse:
        return await self._do_unload()

    async def _do_unload(self) -> UnloadResponse:
        # 口径（任务 55）：用 reserved 而非 allocated 计量 —— empty_cache() 会把
        # 缓存分配器保留的空闲块一并归还驱动，只报 allocated 会系统性低估真实释放量
        # （实测 T1：allocated 30.9GB / reserved 45.4GB）。
        vram_before = self._vram_reserved_mb()

        def _blocking_unload() -> None:
            # 引用断开 → gc → empty_cache（顺序不可变，否则显存不释放）
            self.pipeline = None
            self.transformer = None
            self.comps = None
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass

        await asyncio.to_thread(_blocking_unload)
        self.plan = None
        self._loaded_variant = None
        self._loaded_scheme = None
        self._device = None
        self._load_time = None

        vram_freed = max(vram_before - self._vram_reserved_mb(), 0.0)
        logger.info("unload() 完成：释放显存约 %.1fGB", vram_freed / 1024)
        return UnloadResponse(
            state="idle",
            vram_freed=f"{vram_freed / 1024:.1f}GB" if vram_freed > 0 else None,
        )

    # ============================================================
    # /infer（t2i）
    # ============================================================

    async def _infer_impl(self, request: InferRequest, stream: SSEStream) -> None:
        if self.pipeline is None or self.comps is None:
            raise EngineLoadError(
                "引擎未加载模型（load 未完成或已卸载）。平台 auto-load 应已触发；"
                "若直接调用请先 POST /load。"
            )

        # 1. 参数校验
        try:
            inputs = ImageInputs(**(request.inputs or {}))
        except ValidationError as e:
            logger.warning("infer 参数校验失败：%s", e)
            stream.put_threadsafe(make_error_frame(f"inputs 参数校验失败：{e}", "invalid_inputs"))
            return

        task_id = request.task_id or f"qi21_{int(time.time())}_{id(request) & 0xFFFF:x}"
        gateway_forwarded = bool(request.task_id)

        # 2. 输入图 staging + 模式解析（任务 49：t2i / i2i / edit 三模式）
        #    两条等价通道：InferRequest.images（网关链路） / inputs.images（直连）
        raw_images: list = list(request.images or []) or list(inputs.images or [])
        mode = resolve_mode(inputs.mode, request.workflow, len(raw_images))
        pil_images: list = []
        roles: list[str] = []
        try:
            if raw_images:
                stream.put_threadsafe(make_frame(
                    0.03, "encoding_images", detail=f"载入输入图 {len(raw_images)} 张"))
                pil_images, roles = await asyncio.to_thread(
                    stage_images, raw_images, self.config.input_dir)
            validate_mode(mode, pil_images, roles)
        except ImageStagingError as e:
            logger.warning("输入图/模式校验失败：mode=%s，%s", mode, e)
            stream.put_threadsafe(make_error_frame(f"输入参数错误：{e}", "invalid_inputs"))
            return

        # 2.2 edit 输入归一：独立 mask → 源图高亮标注单图
        #     （QwenImage21Pipeline 无 mask 入参，直接当第 2 张图喂会把 mask 亮度
        #      泄成输出 alpha；任务 50 冒烟实测 C12 踩中）
        # 注：先在「归一前」把原始输入图另存一份到 output（供任务中心回显），
        #     再对 mask 做合并，保证任务中心看到的是用户真正传入的图。
        input_images = await asyncio.to_thread(
            self._save_input_images, task_id, pil_images, roles) if pil_images else []

        if mode != MODE_T2I and "mask" in roles:
            pil_images, roles = await asyncio.to_thread(
                split_edit_inputs, pil_images, roles)

        # 2.3 扩图（outpaint，任务 56）：以「扩展后的画布」为条件图 + 保守提示词前缀。
        #     为什么必须这么做（本轮实测）：仅给更大的输出尺寸而条件图仍是原图时，模型会
        #     **整体重绘**（原图区差异 69.8/255）；改成"扩展画布当输入"后降到 6.7/255。
        expanded_wh: Optional[tuple[int, int]] = None
        if inputs.expand is not None:
            exp = inputs.expand
            err = self._validate_expand(mode, pil_images, roles, exp)
            if err:
                logger.warning("扩图参数校验失败：%s", err)
                stream.put_threadsafe(make_error_frame(err, "invalid_inputs"))
                return
            pil_images = [await asyncio.to_thread(
                expand_canvas, pil_images[0], exp.direction, exp.amount)]
            expanded_wh = pil_images[0].size
            if inputs.width is not None or inputs.height is not None:
                logger.info(
                    "扩图生效：忽略显式尺寸 %s×%s → 以扩展画布 %d×%d 为输出尺寸",
                    inputs.width, inputs.height, *expanded_wh,
                )

        if inputs.denoise is not None:
            # 2.1 无 SD 式 img2img 语义（image= 是条件图，非加噪起点）→ 忽略并告警
            logger.warning(
                "收到 denoise=%.2f：Qwen-Image-2.1 无 SD 式 img2img 语义 → 已忽略"
                "（分析文档 §3.2；不影响本次生成）", inputs.denoise,
            )
        logger.info(
            "推理模式=%s，输入图=%d 张（roles=%s），尺寸策略=%s，透明=%s",
            mode, len(pil_images), describe_roles(roles),
            f"显式 {inputs.width}×{inputs.height}" if inputs.width else "跟随条件图",
            inputs.transparent,
        )

        # 3. variant / scheme 解析（请求 inputs.scheme 或 request.variant）
        variant, scheme_id = self._resolve_request_scheme(inputs, request)

        # 任务中心回显：i2i/edit 未显式给尺寸时，改用条件图尺寸（非 null），
        # 便于前端展示"实际输出尺寸≈条件图尺寸"
        rep_w, rep_h = self._effective_size(inputs, variant, mode)
        if rep_w is None and mode != MODE_T2I and pil_images:
            rep_w, rep_h = pil_images[-1].size
        if expanded_wh is not None:
            # 扩图：回显/输出尺寸都以扩展画布为准（任务 56）
            rep_w, rep_h = expanded_wh

        await self.task_reporting.on_infer_created(
            task_id,
            gateway_forwarded=gateway_forwarded,
            mode=mode,
            request_summary={
                "prompt": inputs.prompt[:200],
                "mode": mode,
                "images": len(pil_images),
                "input_images": input_images,
                "width": rep_w,
                "height": rep_h,
                "seed": inputs.seed,
                "steps": inputs.num_inference_steps,
                "scheme": scheme_id,
            },
            variant=variant,
        )
        self.task_reporting.on_infer_running(task_id, gateway_forwarded=gateway_forwarded)

        try:
            # 3. 档位切换（若请求档位与已加载不符 → 先卸再载）
            if scheme_id and scheme_id != self._loaded_scheme:
                logger.info("请求档位 %s ≠ 已加载 %s → 切换", scheme_id, self._loaded_scheme)
                stream.put_threadsafe(make_frame(0.02, "switching_scheme", detail=f"切换精度档 → {scheme_id}"))
                prev = self.config.default_scheme
                self.config.default_scheme = scheme_id
                try:
                    await self._load_impl(variant)
                finally:
                    self.config.default_scheme = prev

            steps = self._effective_steps(inputs, variant)
            width, height = self._effective_size(inputs, variant, mode)
            if expanded_wh is not None:
                # 扩图：输出尺寸以扩展画布为准（任务 56）
                width, height = expanded_wh

            # 扩图自动提示词前缀（保守指令）——放在任务上报之后注入，
            # 保证任务中心记录的是**用户原文**；仅影响本次管线调用（inputs 是请求级副本）
            if inputs.expand is not None and inputs.expand.auto_prompt:
                prefix = EXPAND_PROMPT_TEMPLATE.format(
                    dir=EXPAND_DIRECTION_LABELS.get(inputs.expand.direction, inputs.expand.direction),
                    prompt="",
                ).rstrip("。") + "。"
                inputs.prompt = prefix + inputs.prompt
                logger.info(
                    "扩图：已注入保守提示词前缀（%s 方向）→ prompt=%s…",
                    inputs.expand.direction, inputs.prompt[:60],
                )

            def _progress(step: int, total: int) -> None:
                frac = 0.1 + 0.85 * (step / max(total, 1))
                stream.put_threadsafe(make_frame(
                    frac, "denoising", detail=f"去噪 {step}/{total}",
                    step=step, total=total,
                ))
                self.task_reporting.on_infer_progress(
                    task_id, gateway_forwarded=gateway_forwarded, progress=frac, stage="denoising",
                )

            stream.put_threadsafe(make_frame(0.05, "encoding", detail="编码提示词"))

            result = await asyncio.to_thread(
                self._run_pipeline,
                inputs, mode, pil_images, width, height, steps, task_id, _progress,
            )

            self._last_used_at = time.time()
            await self.task_reporting.on_infer_succeeded(
                task_id, gateway_forwarded=gateway_forwarded, result=result,
            )
            await stream.put(make_final_frame(result, {
                "total_seconds": result["duration_seconds"],
                "infer_seconds": result["infer_seconds"],
                "mode": mode,
                "scheme": self._loaded_scheme,
            }))
        except Exception as e:  # noqa: BLE001
            logger.exception("推理失败：task_id=%s", task_id)
            self.task_reporting.on_infer_failed(
                task_id, gateway_forwarded=gateway_forwarded,
                error={"type": "engine_error", "message": str(e)},
            )
            stream.put_threadsafe(make_error_frame(f"推理失败：{e}", "engine_error"))

    def _run_pipeline(
        self, inputs: ImageInputs, mode: str, pil_images: list,
        width: Optional[int], height: Optional[int], steps: int,
        task_id: str, on_step,
    ) -> dict:
        """执行 QwenImage21Pipeline（阻塞式，to_thread 调用）。

        mode 差异在此收敛（分析文档 §3.1）：
            t2i  —— 纯文本；尺寸默认 1024²
            edit —— ``image=`` 单张源图（或 ``[源图, mask]`` 两张）；尺寸默认跟随源图
            i2i  —— ``image=`` list(参考图)；尺寸默认跟随末张参考图

        尺寸 width/height 为 None 时**不传**给管线 → 由管线按条件图宽高比推算（官方行为）。
        """
        import torch

        t0 = time.perf_counter()
        gen = None
        if inputs.seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(inputs.seed))

        prompt = inputs.prompt
        if inputs.transparent:
            prompt = TRANSPARENT_PROMPT_TEMPLATE.format(prompt=prompt)
            logger.info("透明背景：已套用官方 RGBA prompt 模板")

        call_kwargs: dict = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "true_cfg_scale": inputs.true_cfg_scale,
            "output_type": "pil",
        }
        if width is not None and height is not None:
            call_kwargs["width"] = width
            call_kwargs["height"] = height
        if mode != MODE_T2I and pil_images:
            # 单张传 PIL；多张传 list（顺序 = 图1/图2… 的指代顺序）
            call_kwargs["image"] = pil_images[0] if len(pil_images) == 1 else list(pil_images)
        if inputs.negative_prompt:
            call_kwargs["negative_prompt"] = inputs.negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen
        if on_step is not None:
            def _cb(pipe, step, timestep, cb_kwargs):
                on_step(step + 1, steps)
                return cb_kwargs
            call_kwargs["callback_on_step_end"] = _cb

        logger.info(
            "调用管线：mode=%s steps=%s cfg=%s size=%s images=%d transparent=%s",
            mode, steps, inputs.true_cfg_scale,
            f"{width}×{height}" if width else "跟随条件图",
            len(pil_images), inputs.transparent,
        )

        out = self.pipeline(**call_kwargs)
        images = getattr(out, "images", None) or out[0]
        infer_seconds = time.perf_counter() - t0

        urls = self._save_images(images, task_id)
        first = images[0] if images else None
        return {
            "image_urls": urls,
            "image_url": urls[0] if urls else "",
            "width": first.size[0] if first is not None else width,
            "height": first.size[1] if first is not None else height,
            "mode": mode,
            "seed": inputs.seed,
            "precision": self._loaded_scheme,
            "infer_seconds": round(infer_seconds, 1),
            "duration_seconds": round(infer_seconds, 1),
        }

    def _save_input_images(self, task_id: str, images: list, roles: list[str]) -> list[dict]:
        """把 staged 输入图另存一份到 output 目录（供任务中心回显），返回 [{url, role}]。

        背景（任务 50 续测）：/infer 直连时输入图只落 ``input_dir`` 临时区，任务中心
        拿不到任何引用 → 编辑任务详情看不到"要编辑的图"。此处在 staging 后另存
        一份到 ``output/<sub>/``（已被引擎静态挂载），把 URL 塞进 request_summary。
        """
        out_dir = self.config.output_root / OUTPUT_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        items: list[dict] = []
        for i, im in enumerate(images):
            name = f"{task_id}_input_{i}.png"
            try:
                im.convert("RGBA").save(out_dir / name)
                items.append({
                    "url": f"/output/{OUTPUT_SUBDIR}/{name}",
                    "role": roles[i] if i < len(roles) and roles[i] else "input",
                })
            except Exception as e:  # noqa: BLE001
                logger.warning("输入图落盘失败（第 %d 张）：%s", i, e)
        logger.info("输入图已另存 %d 张到 output（任务中心回显用）", len(items))
        return items

    def _save_images(self, images: list, task_id: str) -> list[str]:
        """落盘到 output/image/ 并返回 /output/image/... URL。"""
        out_dir = self.config.output_root / OUTPUT_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        urls: list[str] = []
        for i, img in enumerate(images):
            name = f"{task_id}_{i}.png"
            path = out_dir / name
            try:
                img.save(path)
            except Exception as e:  # noqa: BLE001
                logger.error("图像保存失败：%s", e)
                continue
            urls.append(f"/output/{OUTPUT_SUBDIR}/{name}")
            logger.info("输出图像：%s", path)
        return urls

    # ============================================================
    # /status
    # ============================================================

    async def _status_impl(self) -> StatusResponse:
        return StatusResponse(
            state=self.state_machine.state.value,
            loaded_variant=self._loaded_variant,
            vram_used=self._vram_used_str(),
            vram_total=self._vram_total_str(),
            loaded_loras=[],
            last_used_at=self._last_used_at,
            uptime=self.state_machine.uptime,
            extra={
                "scheme": self._loaded_scheme,
                "device": self._device,
                # vram_reserved：进程实际保留的总显存（allocated + 缓存空闲块）。
                # 平台（gateway engine_orchestrator）据「目标引擎自身 reserved」判定
                # 同引擎换档时的可回收显存，避免把自身缓存空闲块误算作他人占用（任务 55）。
                "vram_reserved": self._vram_reserved_str(),
                "dit_dtype": str(self.plan.dit_dtype) if self.plan else None,
                "encoder_dtype": str(self.plan.encoder_dtype) if self.plan else None,
                "plan": self.plan.as_dict() if self.plan else None,
                "load_time": round(self._load_time, 1) if self._load_time else None,
                "hardware": self.schemes.hardware_summary(),
                "schemes": [self._scheme_dict(s) for s in self.schemes.list_schemes()],
            },
        )

    # ============================================================
    # /lora/load、/lora/unload（二期）
    # ============================================================

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        # 保留迁移前的 HTTP 501 语义（原 server.py 把所有异常映射为 501），
        # 与 foleycrafter 等引擎的「不支持 → HTTPException(501)」写法一致
        from fastapi import HTTPException
        raise HTTPException(status_code=501, detail="LoRA 支持为二期功能（任务文档 S3.4 / 分析文档 §7）")

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        from fastapi import HTTPException
        raise HTTPException(status_code=501, detail="LoRA 支持为二期功能（任务文档 S3.4 / 分析文档 §7）")

    # ============================================================
    # 扩展：scheme 目录 / 历史
    # ============================================================

    def list_scheme_dicts(self) -> list[dict]:
        """精度档位目录（WebUI /schemes 端点用）。"""
        return [self._scheme_dict(s) for s in self.schemes.list_schemes()]

    @staticmethod
    def _scheme_dict(s) -> dict:
        return {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "backend_type": s.quantization.backend_type,
            "dit_dtype": s.quantization.dit_dtype,
            "compute_dtype": s.quantization.compute_dtype,
            "encoder_dtype": s.quantization.encoder_dtype,
            "vram_required": s.vram_required,
            "default": s.default,
            "available": s.available,
            "unavailable_reason": s.unavailable_reason,
            "applicable_variants": list(s.applicable_variants),
        }

    async def get_history(self, page: int = 1, page_size: int = 20) -> dict:
        return await fetch_history_page(self.engine_id, page=page, page_size=page_size)

    # ============================================================
    # 辅助
    # ============================================================

    def _resolve_request_scheme(self, inputs: ImageInputs, request: InferRequest) -> tuple[str, Optional[str]]:
        """解析本次请求的 (variant, scheme_id)。

        variant：request.variant（gateway model 字段）> 已加载 > default。
        scheme：inputs.scheme 显式 > 由 variant 的默认适用档推导。
        """
        variant = request.variant or self._loaded_variant or self.default_variant or ""
        scheme_id = inputs.scheme
        if scheme_id is None and variant:
            s = self.schemes.auto_select_scheme(variant)
            scheme_id = s.id if s else None
        return variant, scheme_id

    def _effective_steps(self, inputs: ImageInputs, variant: str) -> int:
        if inputs.num_inference_steps is not None:
            return inputs.num_inference_steps
        if "turbo" in (variant or "").lower():
            return 8
        return 40

    @staticmethod
    def _validate_expand(
        mode: str, pil_images: list, roles: list[str], exp: ExpandSpec,
    ) -> Optional[str]:
        """扩图参数校验（返回可读错误串；None = 校验通过）。任务 56。"""
        if mode != MODE_EDIT:
            return (
                f"扩图仅支持「图像编辑」模式（单张源图）；当前模式={mode}。"
                f"请切到「图像编辑」后重试"
            )
        if "mask" in roles:
            return (
                "扩图与「独立蒙版」不能同时使用：蒙版会先被合并成高亮标注图，"
                "而扩图改的是画布几何，两者语义冲突 → 请二选一"
            )
        if len(pil_images) != 1:
            return f"扩图需要且仅需要 1 张源图（当前 {len(pil_images)} 张）"
        w, h = pil_images[0].size
        nw, nh = expanded_size(w, h, exp.direction, exp.amount)
        if nw > SIZE_MAX or nh > SIZE_MAX:
            return (
                f"扩图后画布 {nw}×{nh} 超出上限 {SIZE_MAX}×{SIZE_MAX}"
                f"（源图 {w}×{h}，方向 {exp.direction}，扩幅 {exp.amount}px）→ 请减小扩幅"
            )
        return None

    def _effective_size(
        self, inputs: ImageInputs, variant: str, mode: str = MODE_T2I,
    ) -> tuple[Optional[int], Optional[int]]:
        """解析输出尺寸。

        t2i：默认 1024²；**edit / i2i：留空 → 交给管线按条件图宽高比推算**（官方行为）。
        显式给出时直接用（pydantic 已保证 32 对齐 + 256–2048 范围）。

        注：扩图（`inputs.expand`）的输出尺寸在 `_infer_impl` 里以**扩展画布**覆盖，
        不走本函数；显式 width/height 在扩图生效时被忽略（引擎记日志说明）。
        """
        if inputs.width is not None and inputs.height is not None:
            return inputs.width, inputs.height
        if mode == MODE_T2I:
            return 1024, 1024
        return None, None

    def _read_default_variant_from_manifest(self, manifest_path: Path) -> Optional[str]:
        try:
            import yaml
            data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            return (data.get("models", {}) or {}).get("default_variant")
        except Exception as e:  # noqa: BLE001
            logger.warning("读取 manifest default_variant 失败：%s", e)
            return None

    def _vram_used_mb(self) -> float:
        """已分配显存（活着的张量）。"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**2
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    def _vram_used_str(self) -> str:
        return f"{self._vram_used_mb() / 1024:.1f}GB"

    def _vram_reserved_mb(self) -> float:
        """已保留显存（已分配 + 缓存分配器空闲块）。

        口径说明（任务 55 实测）：
            memory_allocated() 只含"活着的张量"；而 caching allocator 向驱动
            申请后未分配出去的空闲块由 memory_reserved() 体现。
            本机 T1 载入时 allocated=30.9GB 而 reserved≈46.7GB（差 ≈15.8GB），
            差额对**同进程下一次 load()** 可直接复用，但对驱动/nvidia-smi 表现为"已占用"。
            平台需据此判定"引擎自身可回收显存"，否则同引擎换档会被误判显存不足。
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_reserved() / 1024**2
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    def _vram_reserved_str(self) -> str:
        return f"{self._vram_reserved_mb() / 1024:.1f}GB"

    def _vram_total_str(self) -> Optional[str]:
        try:
            import torch
            if torch.cuda.is_available():
                return f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        except Exception:  # noqa: BLE001
            pass
        return None
