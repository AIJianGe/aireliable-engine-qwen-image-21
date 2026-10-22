"""HTTP 服务契约 + 原生 WebUI（根路径 307 → /ui/，子路径接口必须用根绝对路径）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engines.qwen_image_21_feeb8130 import server
from engines.qwen_image_21_feeb8130.config import EngineConfig

ENGINE_DIR = Path(__file__).resolve().parents[1]
WEBUI_DIR = ENGINE_DIR / "webui"

SDK_ENDPOINTS = [
    "/health", "/capabilities", "/status", "/load", "/unload", "/infer",
    "/lora/load", "/lora/unload",
]
EXTRA_ENDPOINTS = ["/schemes", "/history"]


@pytest.fixture()
def app(tmp_path: Path):
    cfg = EngineConfig(
        project_root=tmp_path,
        models_root=tmp_path / "models",
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
    )
    return server.create_app(cfg)


def test_webui_files_exist():
    assert (WEBUI_DIR / "index.html").exists(), "webui/index.html 缺失（平台 iframe 会显示 JSON）"
    assert (WEBUI_DIR / "app.js").exists()
    assert (WEBUI_DIR / "style.css").exists()


def test_root_redirects_307_to_ui(app):
    client = TestClient(app)
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/ui/"


def test_webui_mounted(app):
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/ui" in paths
    assert "/output" in paths


def test_all_sdk_endpoints_registered(app):
    paths = {getattr(r, "path", None) for r in app.routes}
    for ep in SDK_ENDPOINTS + EXTRA_ENDPOINTS:
        assert ep in paths, f"缺少端点 {ep}"


def test_cors_middleware_present(app):
    names = [m.cls.__name__ for m in app.user_middleware]
    assert "CORSMiddleware" in names


def test_webui_uses_absolute_api_paths():
    """子路径页面（/ui/）的接口调用必须写成根绝对路径，否则解析成 /ui/xxx → 404。"""
    import re

    js = (WEBUI_DIR / "app.js").read_text(encoding="utf-8")
    # UI 实际调用的端点
    for ep in ("capabilities", "schemes", "status", "load", "infer", "history"):
        assert f"/{ep}" in js, f"app.js 未调用 /{ep}"
    # 所有字面量请求路径（fetch/jget 的第一个字符串实参）必须以 "/" 开头
    targets = re.findall(r"""(?:jget|fetch)\(\s*(["'])([^"']*)\1""", js)
    assert targets, "app.js 未发现任何字面量请求路径"
    for _quote, path in targets:
        assert path.startswith("/"), f"app.js 出现相对 API 路径：{path!r}（应为 /{path.lstrip('/')}）"


def test_index_html_references_assets():
    html = (WEBUI_DIR / "index.html").read_text(encoding="utf-8")
    assert "app.js" in html
    assert "style.css" in html


def test_openapi_doc_title(app):
    assert "Qwen-Image2.1" in app.title


# ===== 任务 49：三模式工作台 UI =====

def test_webui_has_three_mode_tabs():
    html = (WEBUI_DIR / "index.html").read_text(encoding="utf-8")
    for m in ('data-mode="t2i"', 'data-mode="i2i"', 'data-mode="edit"'):
        assert m in html, f"index.html 缺少模式切换项 {m}"
    assert "文生图" in html and "图生图" in html and "图像编辑" in html


def test_webui_has_upload_and_local_edit_ui():
    html = (WEBUI_DIR / "index.html").read_text(encoding="utf-8")
    for token in ("dropzone", "file-input", "thumbs", "mask-canvas", "local-mode",
                  "transparent", "compare", "reuse-btn", "follow-size"):
        assert token in html, f"index.html 缺少控件 {token}"


def test_webui_mode_logic_in_appjs():
    js = (WEBUI_DIR / "app.js").read_text(encoding="utf-8")
    for token in ("setMode", "buildImagesPayload", "composeMarked", "composeMask",
                  "pointerdown", "b64_json", "role", "dataTransfer", "clipboardData"):
        assert token in js, f"app.js 缺少逻辑 {token}"


def test_webui_validation_feedback_is_prominent():
    """v2.1：校验失败必须走 setErr（toast + 红字），不能只写小字 #hint。"""
    js = (WEBUI_DIR / "app.js").read_text(encoding="utf-8")
    css = (WEBUI_DIR / "style.css").read_text(encoding="utf-8")
    for token in ("setErr", "showToast", "toast-wrap", "hint-error", "setImageHint", "flash-error"):
        assert token in js or token in css, f"缺少显眼反馈实现 {token}"
    # 图生图张数校验必须使用 setErr + 图片区红框锚点
    assert "图生图需要 2–" in js and 'setErr(msg, $("image-block"))' in js
    for token in (".hint-error", ".toast", ".toast-wrap", ".flash-error"):
        assert token in css, f"style.css 缺少样式 {token}"


def test_webui_single_scheme_entry_autopairs():
    """任务 51 · 方案 A：精度档收为只读标签，由模型自动带出；请求必带成对 (variant, scheme)。"""
    html = (WEBUI_DIR / "index.html").read_text(encoding="utf-8")
    js = (WEBUI_DIR / "app.js").read_text(encoding="utf-8")
    css = (WEBUI_DIR / "style.css").read_text(encoding="utf-8")
    # 只读精度标签取代原精度下拉
    assert 'id="scheme-badge"' in html and 'id="scheme-select"' not in html
    for token in ("schemeForVariant", "resolvePair", "applyPair", "renderVariantOptions", "humanize"):
        assert token in js, f"app.js 缺少自动配对逻辑 {token}"
    # 模式变更清除上次提示（R10）
    assert "clearFeedback" in js
    # 不可用档位灰显 + 只读标签样式
    for token in (".scheme-badge", ".scheme-badge-wrap"):
        assert token in css, f"style.css 缺少样式 {token}"


def test_webui_mode_first_control_visibility():
    """三模式为主轴：t2i 隐藏参考图/局部编辑/扩图；i2i/edit 隐藏透明。

    任务 56 变更：**尺寸区不再对 i2i/edit 整行隐藏**（后端一直支持显式宽高，此前只是
    被 `size-row.hidden = needImg` 挡住）→ 改为「跟随条件图」复选（`follow-lbl`）控制；
    并新增扩图控件块（仅 edit）。
    """
    js = (WEBUI_DIR / "app.js").read_text(encoding="utf-8")
    html = (WEBUI_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="transparent-lbl"' in html                      # 透明开关可整体隐藏
    assert 'id="expand-block"' in html and 'id="follow-size"' in html
    for token in ('$("transparent-lbl").hidden = needImg',
                  '$("size-row").hidden = false',
                  '$("ratio-row").hidden = false',
                  '$("follow-lbl").hidden = !needImg',
                  '$("mask-block").hidden = m !== "edit"',
                  '$("expand-block").hidden = m !== "edit"',
                  '$("image-block").hidden = !needImg'):
        assert token in js, f"setMode 缺少显隐规则：{token}"
    # 上传时不再静默丢图（改由 buildImagesPayload 提交时取第 1 张）
    assert 'if (mode === "edit") images = [];' not in js


def test_appjs_sends_mode_field():
    """三模式通过 inputs.mode 传给引擎。"""
    js = (WEBUI_DIR / "app.js").read_text(encoding="utf-8")
    assert "mode," in js or "mode:" in js
    assert "'i2i'" in js or '"i2i"' in js


def test_appjs_size_snap_rounds_up():
    """任务 56：尺寸吸附必须**向上取整**（与后端 align_up 同口径）。

    实测教训（T-E6）：原实现 `Math.round(v/32)*32` 把用户输入的 1000 变成 992
    （**比用户要的更小**），与后端"向上对齐"语义相反 → 已改为 Math.ceil。
    """
    js = (WEBUI_DIR / "app.js").read_text(encoding="utf-8")
    assert "Math.ceil(Number(v) / SIZE_ALIGN) * SIZE_ALIGN" in js, "尺寸吸附未使用向上取整"
    assert "snapAlign" in js


def test_appjs_default_variant_prefers_default_scheme():
    """任务 56-V2：档位默认必须以「默认精度档」的模型为准，而非 capabilities 列表首项。

    实测教训：本机 /capabilities 首项是 bf16（T1），页面若按首项默认选中，
    会"一开页就是 FP 档"，首次生成静默触发整档切换（~30GB 重载 + 质量档不符基线）。
    """
    js = (WEBUI_DIR / "app.js").read_text(encoding="utf-8")
    assert "schemes.find((s) => s.default)" in js, "未按默认精度档决定默认模型"
    assert "applicable_variants" in js and "defVariant ||" in js
