"""Qwen-Image-2.1 图像引擎 HTTP 服务（engine_id: qwen_image_21_feeb8130）。

8 个标准端点 / CORS / 日志端点 / 静态挂载 / 根路径跳转由 SDK 骨架
（``build_engine_app``）提供；本文件只保留引擎特有路由。

端点：
    GET  /health  /capabilities  /status          （骨架）
    GET  /models    模型清单（与平台「模型管理」同口径，骨架）
    GET  /sdk/model-inventory.js  WebUI 共享模型清单组件（骨架）
    POST /load  /unload  /infer(SSE)  /lora/*     （骨架；LoRA 为二期 → 501）
    GET  /schemes   4 档精度目录（可用性 / 硬件适配）
    GET  /history   代理 gateway 任务中心
    GET  /info      引擎信息（恒可用）
    GET  /output/*  输出图静态服务
    GET|POST /log_level（骨架）
    GET  /ui/*      WebUI
    GET  /          307 → /ui/

启动：python -m engines.qwen_image_21_feeb8130.server --port 7009
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException

from aireliable_engine import build_engine_app, run_engine_server

from .config import EngineConfig, parse_args
from .engine import QwenImage21Engine

logger = logging.getLogger(__name__)

_WEBUI_DIR = Path(__file__).parent / "webui"


# ===== 全局引擎实例 =====
# 由 SDK 骨架的 startup_hook 绑定，供本文件的路由使用
_engine: Optional[QwenImage21Engine] = None
_config: Optional[EngineConfig] = None


def get_engine() -> QwenImage21Engine:
    """获取引擎实例。"""
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


def get_config() -> EngineConfig:
    """获取配置。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config


# ===== Lifespan 钩子（由 SDK 骨架调用）=====

async def _on_startup(engine: QwenImage21Engine) -> None:
    """启动钩子：绑定全局实例。"""
    global _engine

    _engine = engine
    logger.info(
        "引擎实例化完成：state=%s default_variant=%s schemes=%s",
        engine.state_machine.state.value, engine.default_variant,
        [s["id"] for s in engine.list_scheme_dicts()],
    )


async def _on_shutdown(engine: QwenImage21Engine) -> None:
    """关闭钩子：卸载模型并清空全局引用。"""
    global _engine, _config

    logger.info("===== Qwen-Image-2.1 图像引擎关闭 =====")
    try:
        if engine.pipeline is not None:
            try:
                await engine._unload_impl()
            except Exception as e:  # noqa: BLE001
                logger.warning("关闭时卸载模型失败：%s", e)
    finally:
        _engine = None
        _config = None


# ===== 引擎特有路由 =====

router = APIRouter()


@router.get("/schemes")
async def schemes() -> dict:
    """4 档精度档位目录（WebUI 精度下拉数据源）。"""
    eng = get_engine()
    return {
        "schemes": eng.list_scheme_dicts(),
        "hardware": eng.schemes.hardware_summary(),
        "default": eng.config.default_scheme,
    }


@router.get("/history")
async def history(page: int = 1, page_size: int = 20) -> dict:
    """生成历史（最新在前，gateway 任务中心代理视图）。"""
    if page < 1 or page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="page ≥ 1 且 1 ≤ page_size ≤ 100")
    return await get_engine().get_history(page=page, page_size=page_size)


@router.get("/info", include_in_schema=False)
async def info() -> dict:
    """引擎信息（恒可用）。

    任务 58 走网络实测（C5.3）发现本引擎缺 ``/info`` → 脚本 / 排障 / MCP 发现端点
    拿不到稳定的 JSON 入口（根路径一律 307 跳 WebUI）。补齐后与
    ``video_up_enlarge_98782fcd`` 同范式（平台规则 §六.12）。
    """
    cfg = get_config()
    eng = get_engine()
    return {
        "engine_id": cfg.engine_id,
        "engine_name": cfg.engine_name,
        "capability": cfg.capability,
        "type": cfg.engine_type,
        "version": "0.1.0",
        "default_scheme": cfg.default_scheme,
        "schemes": [s["id"] for s in eng.list_scheme_dicts()],
        "webui": "/ui/",
        "endpoints": [
            "GET /health", "GET /capabilities", "GET /schemes", "GET /status",
            "GET /models", "GET /sdk/model-inventory.js",
            "POST /load", "POST /unload", "POST /infer",
            "POST /lora/load", "POST /lora/unload",
            "GET /history", "GET /output/*", "GET|POST /log_level",
            "GET /ui/", "GET /info",
        ],
        "note": "WebUI: GET /ui/；引擎信息: GET /info",
    }


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config

    if config is None:
        config = parse_args()
    _config = config

    return build_engine_app(
        QwenImage21Engine,
        config=config,
        title="Qwen-Image2.1 Engine",
        description=(
            "AIReliable 平台的 Qwen-Image-2.1 图像引擎（t2i，4 档精度 T1-T4，"
            "运行时按显卡自适应 dtype）"
        ),
        version="0.1.0",
        static_dirs={"/ui": _WEBUI_DIR, "/output": config.output_root},
        extra_routers=[router],
        startup_hook=_on_startup,
        shutdown_hook=_on_shutdown,
    )


# ===== 入口 =====

# 模块级 app 实例（uvicorn 直接引用）
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global app

    config = parse_args()
    app = create_app(config)
    run_engine_server(app, config=config)


if __name__ == "__main__":
    main()
