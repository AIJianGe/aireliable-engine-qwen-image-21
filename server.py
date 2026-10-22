"""Qwen-Image-2.1 图像引擎 HTTP 服务（FastAPI，engine_id: qwen_image_21_feeb8130）。

标准端点（SDK 契约）：
    GET  /health  /capabilities  /status
    POST /load  /unload  /infer(SSE)  /lora/load  /lora/unload
扩展端点（WebUI / 平台对接）：
    GET  /schemes  （4 档精度目录：可用性 / 硬件适配）
    GET  /history  （代理 gateway 任务中心）
    GET/POST /log_level（SDK 提供）
    GET  /output/* （输出图静态服务）

启动：python -m engines.qwen_image_21_feeb8130.server --port 7009
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from aireliable_engine import (
    HealthResponse,
    InferRequest,
    LoadRequest,
    LoraLoadRequest,
    LoraUnloadRequest,
    register_log_level_endpoint,
    setup_engine_logging,
)

from .config import EngineConfig, parse_args
from .engine import QwenImage21Engine

logger = logging.getLogger(__name__)

_engine: Optional[QwenImage21Engine] = None
_config: Optional[EngineConfig] = None


def get_engine() -> QwenImage21Engine:
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


# ===== Lifespan =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _config
    if _config is None:
        _config = parse_args()

    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )
    logger.info("===== Qwen-Image-2.1 图像引擎启动（qwen_image_21_feeb8130）=====")
    logger.info("配置: %s", _config.to_dict())

    _engine = QwenImage21Engine(_config)
    logger.info(
        "引擎实例化完成：state=%s default_variant=%s schemes=%s",
        _engine.state_machine.state.value, _engine.default_variant,
        [s["id"] for s in _engine.list_scheme_dicts()],
    )
    yield
    logger.info("===== Qwen-Image-2.1 图像引擎关闭 =====")
    if _engine is not None and _engine.pipeline is not None:
        try:
            await _engine._unload_impl()
        except Exception as e:  # noqa: BLE001
            logger.warning("关闭时卸载模型失败：%s", e)
    _engine = None


def create_app(config: EngineConfig | None = None) -> FastAPI:
    global _config
    if config is not None:
        _config = config

    app = FastAPI(
        title="Qwen-Image2.1 Engine",
        description=(
            "AIReliable 平台的 Qwen-Image-2.1 图像引擎（t2i，4 档精度 T1-T4，"
            "运行时按显卡自适应 dtype）"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_endpoints(app)

    @app.get("/info", include_in_schema=False)
    async def _info() -> dict:
        """引擎信息（恒可用）。

        任务 58 走网络实测（C5.3）发现本引擎缺 ``/info`` → 脚本 / 排障 / MCP 发现端点
        拿不到稳定的 JSON 入口（根路径一律 307 跳 WebUI）。补齐后与
        ``video_up_enlarge_98782fcd`` 同范式（平台规则 §六.12）。
        """
        cfg = _config
        eng = get_engine()
        return {
            "engine_id": cfg.engine_id,
            "engine_name": cfg.engine_name,
            "capability": cfg.capability,
            "type": cfg.engine_type,
            "version": app.version,
            "default_scheme": cfg.default_scheme,
            "schemes": [s["id"] for s in eng.list_scheme_dicts()],
            "webui": "/ui/",
            "endpoints": [
                "GET /health", "GET /capabilities", "GET /schemes", "GET /status",
                "POST /load", "POST /unload", "POST /infer",
                "POST /lora/load", "POST /lora/unload",
                "GET /history", "GET /output/*", "GET|POST /log_level",
                "GET /ui/", "GET /info",
            ],
            "note": "WebUI: GET /ui/；引擎信息: GET /info",
        }

    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/", status_code=307)

    if _config is not None:
        app.mount("/output", StaticFiles(directory=str(_config.output_root)), name="output")
        logger.info("输出目录挂载: /output -> %s", _config.output_root)

    _mount_webui(app)
    return app


def _mount_webui(app: FastAPI) -> None:
    webui_dir = Path(__file__).parent / "webui"
    if webui_dir.is_dir() and (webui_dir / "index.html").exists():
        app.mount("/ui", StaticFiles(directory=str(webui_dir), html=True), name="webui")
        logger.info("WebUI 挂载: /ui -> %s", webui_dir)
    else:
        logger.info("WebUI 未挂载（webui/index.html 不存在）")


def _register_endpoints(app: FastAPI) -> None:

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return await get_engine().health()

    @app.get("/capabilities")
    async def capabilities() -> dict:
        eng = get_engine()
        resp = await eng.capabilities()
        result = resp.model_dump(exclude={"variants"})
        result["variants"] = [v.model_dump() for v in resp.variants]
        # 任务 49：三模式能力声明（SDK CapabilitiesResponse 无 modes 字段 → 此处注入）
        result.update(eng.capabilities_extra())
        return result

    @app.get("/schemes")
    async def schemes() -> dict:
        """4 档精度档位目录（WebUI 精度下拉数据源）。"""
        eng = get_engine()
        return {
            "schemes": eng.list_scheme_dicts(),
            "hardware": eng.schemes.hardware_summary(),
            "default": eng.config.default_scheme,
        }

    @app.post("/load")
    async def load(request: LoadRequest) -> dict:
        try:
            return (await get_engine().load(request)).model_dump()
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:  # noqa: BLE001
            logger.exception("加载失败: variant=%s", request.variant)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/unload")
    async def unload() -> dict:
        try:
            return (await get_engine().unload()).model_dump()
        except Exception as e:  # noqa: BLE001
            logger.exception("卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/infer")
    async def infer(request: InferRequest):
        return StreamingResponse(
            get_engine().infer(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @app.get("/status")
    async def status() -> dict:
        return (await get_engine().status()).model_dump()

    @app.post("/lora/load")
    async def lora_load(request: LoraLoadRequest) -> dict:
        try:
            return (await get_engine().load_lora(request)).model_dump()
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("LoRA 加载失败: %s", request.name)
            raise HTTPException(status_code=501, detail=str(e))

    @app.post("/lora/unload")
    async def lora_unload(request: LoraUnloadRequest) -> dict:
        try:
            return (await get_engine().unload_lora(request)).model_dump()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=501, detail=str(e))

    @app.get("/history")
    async def history(page: int = 1, page_size: int = 20) -> dict:
        if page < 1 or page_size < 1 or page_size > 100:
            raise HTTPException(status_code=400, detail="page ≥ 1 且 1 ≤ page_size ≤ 100")
        return await get_engine().get_history(page=page, page_size=page_size)

    register_log_level_endpoint(app)


app: Optional[FastAPI] = None


def main() -> None:
    global _config, app
    _config = parse_args()
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )
    logger.info("启动 Qwen-Image-2.1 图像引擎: host=%s, port=%d", _config.host, _config.port)
    app = create_app(_config)
    import uvicorn
    uvicorn.run(app, host=_config.host, port=_config.port, log_level=_config.log_level.lower())


if __name__ == "__main__":
    main()
