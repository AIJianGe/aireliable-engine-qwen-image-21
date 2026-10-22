"""任务中心对接（Qwen-Image2.1 引擎）。

gateway ``data/tasks.db`` 为任务元数据唯一真相源；引擎不自建历史，
``/history`` 代理查询 gateway（不可达降级）。

上报分工：
    - gateway 转发链路（request.task_id 非空）：仅报 created（gateway 维护进度/终态）
    - 直连链路（WebUI）：报全生命周期 created → running → progress → succeeded/failed

task_type 映射：t2i → image_generation（一期仅文生图）。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import httpx

from aireliable_engine import PlatformTaskReporter

logger = logging.getLogger(__name__)

TASK_TYPE_BY_MODE: dict[str, str] = {"t2i": "image_generation"}

_DEFAULT_GATEWAY_URL = "http://127.0.0.1:8000"


def _gateway_url() -> str:
    return (os.environ.get("AIRELIABLE_GATEWAY_URL", "") or _DEFAULT_GATEWAY_URL).rstrip("/")


class QwenImage21TaskReporting:
    """任务上报薄封装（惰性初始化，容错优先，不阻断推理）。"""

    def __init__(self, engine_id: str):
        self._engine_id = engine_id
        self._reporter: Optional[PlatformTaskReporter] = None

    async def _ensure_reporter(self) -> Optional[PlatformTaskReporter]:
        if self._reporter is None:
            try:
                self._reporter = PlatformTaskReporter(engine_id=self._engine_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("任务上报器初始化失败（不阻断推理）: %s", e)
        return self._reporter

    async def on_infer_created(
        self, task_id: str, *, gateway_forwarded: bool, mode: str,
        request_summary: dict[str, Any], variant: str,
    ) -> None:
        reporter = await self._ensure_reporter()
        if reporter is None:
            return
        try:
            if gateway_forwarded:
                await reporter.report_created(task_id=task_id, variant=variant)
            else:
                await reporter.report_created(
                    task_id=task_id,
                    request_summary=request_summary,
                    variant=variant,
                    task_type=TASK_TYPE_BY_MODE.get(mode, "image_generation"),
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("report_created 失败（不阻断推理）: %s", e)

    def on_infer_running(self, task_id: str, *, gateway_forwarded: bool) -> None:
        if gateway_forwarded or self._reporter is None:
            return
        self._reporter.fire_report_running(task_id)

    def on_infer_progress(self, task_id: str, *, gateway_forwarded: bool, progress: float, stage: str) -> None:
        if gateway_forwarded or self._reporter is None:
            return
        self._reporter.fire_report_progress(task_id, progress=progress, stage=stage)

    async def on_infer_succeeded(self, task_id: str, *, gateway_forwarded: bool, result: dict[str, Any]) -> None:
        if gateway_forwarded:
            return
        reporter = await self._ensure_reporter()
        if reporter is None:
            return
        try:
            await reporter.report_succeeded(task_id=task_id, result=result)
        except Exception as e:  # noqa: BLE001
            logger.warning("report_succeeded 失败（不阻断推理）: %s", e)

    def on_infer_failed(self, task_id: str, *, gateway_forwarded: bool, error: dict[str, Any]) -> None:
        if gateway_forwarded or self._reporter is None:
            return
        self._reporter.fire_report_failed(task_id, error=error)


async def fetch_history_page(engine_id: str, page: int = 1, page_size: int = 20) -> dict:
    """代理查询 gateway 任务中心，映射为历史视图结构。降级返回空列表。"""
    offset = (page - 1) * page_size
    try:
        async with httpx.AsyncClient(
            base_url=_gateway_url(),
            timeout=httpx.Timeout(3.0, connect=1.5),
            trust_env=False,
        ) as client:
            resp = await client.get("/v1/tasks", params={
                "engine_id": engine_id,
                "status": "succeeded",
                "limit": page_size,
                "offset": offset,
            })
            resp.raise_for_status()
            data = resp.json()
            tasks = data.get("tasks") or []
            items = [_map_task_to_history(t) for t in tasks if t.get("image_url")]
            return {"items": items, "total": data.get("total", 0), "page": page, "page_size": page_size}
    except Exception as e:  # noqa: BLE001
        logger.warning("历史视图查询 gateway 失败（降级为空列表，原因：%s）", e)
        return {"items": [], "total": 0, "page": page, "page_size": page_size, "degraded": True}


def _map_task_to_history(t: dict) -> dict:
    """任务记录 → HistoryItem 视图形状（沿用 7007 引擎的映射口径）。"""
    rs = t.get("request_summary") or {}
    return {
        "task_id": t.get("task_id", ""),
        "mode": rs.get("mode", "t2i"),
        "prompt": rs.get("prompt", ""),
        "image_url": t.get("image_url", ""),
        "thumbnail_url": None,
        "width": rs.get("width"),
        "height": rs.get("height"),
        "seed": rs.get("seed"),
        "precision": t.get("variant_id"),
        "loras": [],
        "created_at": t.get("created_at", 0.0),
        "duration_seconds": t.get("duration_seconds", 0.0) or 0.0,
    }
