"""输入图 staging + 模式解析（任务 49 · P1/P2，分析文档 §4.1/§3.1）。

职责：
    1. 把 ``InferRequest.images``（``ImageInput``：url / b64_json / path + role）
       或 ``inputs.images``（dict 形式）统一落盘为 PIL 图（**保留 RGBA**）
    2. 解析推理模式（t2i / i2i / edit）：显式 mode > request.workflow > 按图数推断
    3. 校验各模式的图片张数（越界抛 ``ImageStagingError`` → SSE ``invalid_inputs``）

铁律（分析文档 §3.2）：Qwen-Image-2.1 的 ``image=`` 是**条件图**（"看着源图重画"），
**没有** SD 式 ``denoise/strength`` 语义；mask 也**不是** latent mask，而是"把 mask 当
第二张图"看（官方三法：圈选 / 涂抹标注 / 原图+独立 mask）。
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageFilter
import numpy as np

# 尺寸对齐契约（管线 multiple_of = 32；见 schemas 模块说明）
from .schemas import SIZE_ALIGN

logger = logging.getLogger(__name__)

# ===== 常量（分析文档 §2.2「多图数量」经验）=====
MAX_REFERENCE_IMAGES = 10      # 官方上限（硬）
SOFT_REFERENCE_IMAGES = 4      # 社区最佳 1–3，>4 指令遵循度下降（软提示）
MIN_SIDE = 256                 # 输入图最短边下限
MAX_SIDE = 8192                # 输入图最长边上限
MAX_IMAGE_BYTES = 20 * 1024**2  # 单图 20MB（含 base64 解码后）

# ===== 模式 =====
MODE_T2I = "t2i"
MODE_I2I = "i2i"
MODE_EDIT = "edit"
SUPPORTED_MODES = (MODE_T2I, MODE_I2I, MODE_EDIT)

# 别名归一（gateway workflow / 平台任务类型 / 用户口语）
_MODE_ALIASES: dict[str, str] = {
    "t2i": MODE_T2I, "text_to_image": MODE_T2I, "image_generation": MODE_T2I,
    "i2i": MODE_I2I, "image_to_image": MODE_I2I, "img2img": MODE_I2I,
    "edit": MODE_EDIT, "image_editing": MODE_EDIT, "image_edit": MODE_EDIT,
}

# 局部编辑三法（供 /capabilities 声明）
LOCAL_EDIT_METHODS = ("circle", "paint", "mask")


class ImageStagingError(ValueError):
    """输入图 staging / 模式校验失败（调用方转 SSE invalid_inputs 帧）。"""


# ============================================================
# 模式解析
# ============================================================

def normalize_mode(value: Optional[str]) -> Optional[str]:
    """把别名归一为 t2i / i2i / edit；无法识别返回 None。"""
    if not value:
        return None
    return _MODE_ALIASES.get(str(value).strip().lower())


def resolve_mode(
    explicit: Optional[str],
    workflow: Optional[str],
    n_images: int,
) -> str:
    """解析推理模式。

    优先级：``inputs.mode`` > ``request.workflow`` > 按图片张数推断
    （0 → t2i；1 → edit；≥2 → i2i）。
    """
    for cand in (explicit, workflow):
        m = normalize_mode(cand)
        if m is not None:
            return m
    return MODE_T2I if n_images == 0 else (MODE_EDIT if n_images == 1 else MODE_I2I)


def validate_mode(mode: str, images: list[Any], roles: list[str]) -> None:
    """校验模式与图片张数是否自洽（不自洽抛 ImageStagingError）。"""
    n = len(images)
    if mode == MODE_T2I:
        if n:
            raise ImageStagingError(
                f"mode=t2i（文生图）不接受输入图，但收到 {n} 张（roles={roles}）。"
                f"请改用 mode=i2i/edit，或不传 images。"
            )
        return
    if mode == MODE_EDIT:
        if n < 1:
            raise ImageStagingError("mode=edit（图像编辑）需要至少 1 张源图（images 为空）")
    elif mode == MODE_I2I:
        if n < 2:
            raise ImageStagingError(
                f"mode=i2i（图生图多图参考）需要 2–{MAX_REFERENCE_IMAGES} 张参考图（收到 {n} 张）。"
                f"单图编辑请用 mode=edit。"
            )
    else:
        raise ImageStagingError(f"未知 mode={mode!r}（合法：{SUPPORTED_MODES}）")

    if n > MAX_REFERENCE_IMAGES:
        raise ImageStagingError(
            f"输入图超上限：{n} > {MAX_REFERENCE_IMAGES}（官方上限，"
            f"vision token 会爆炸且指令遵循度下降）"
        )
    if n > SOFT_REFERENCE_IMAGES:
        logger.warning(
            "输入图 %d 张 > 建议上限 %d：社区实测 >%d 张指令遵循度下降，已按官方上限放行",
            n, SOFT_REFERENCE_IMAGES, SOFT_REFERENCE_IMAGES,
        )


def describe_roles(roles: list[str]) -> dict[str, int]:
    """role 计数（日志用）。"""
    out: dict[str, int] = {}
    for r in roles:
        out[r or "unspecified"] = out.get(r or "unspecified", 0) + 1
    return out


def merge_mask_overlay(source: Image.Image, mask: Image.Image) -> Image.Image:
    """把「原图 + 独立 mask」合并为「原图 + 高亮标注」单图（局部编辑 mask 法）。

    动机（任务 50 冒烟实测）：``QwenImage21Pipeline`` 的 ``image=`` 无 mask 入参，
    把 mask 当**第 2 张条件图**喂入时，管线会走 RGBA 条件分支，输出 alpha 退化成
    mask 亮度（mask 区域变成透明洞），目标编辑（加 logo 等）完全不生效。

    改为把 mask 区域以**半透明红高亮 + 描边**叠加到源图上，只传单张条件图 —— 与
    已验证可用的"圈选合成"法同源（模型能理解标注并抹去标注、按指令编辑）。
    """
    src = source.convert("RGB")
    m = mask.convert("L")
    if m.size != src.size:
        m = m.resize(src.size, Image.NEAREST)

    arr = np.asarray(src, dtype=np.float32).copy()
    region = (np.asarray(m, dtype=np.float32) >= 128.0).astype(np.float32)
    tint = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    alpha = 0.40
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1.0 - alpha * region) + tint[c] * alpha * region
    out = Image.fromarray(arr.astype("uint8"), "RGB")

    # 描边（边界更明确）：MaxFilter 膨胀 3px 后取新增像素（不依赖 scipy）
    binmask = m.point(lambda v: 255 if v >= 128 else 0, mode="L")
    outline = (np.asarray(binmask.filter(ImageFilter.MaxFilter(7))) > 0) & (
        np.asarray(binmask) == 0)
    if outline.any():
        e = np.asarray(out).copy()
        e[outline] = np.array([190, 0, 0], dtype=np.uint8)
        out = Image.fromarray(e, "RGB")
    return out


def split_edit_inputs(
    images: list[Image.Image], roles: list[str],
) -> tuple[list[Image.Image], list[str]]:
    """edit 模式输入归一：把独立 mask 合并进源图，返回单/多张条件图。

    - 无 mask：原样返回（单图源 / 多图参考）
    - 有 mask（role="mask"）：源图叠加 mask 高亮，丢弃 mask 张
    """
    if "mask" not in roles:
        return images, roles
    src_idx = next((i for i, r in enumerate(roles) if r == "source"), 0)
    merged = merge_mask_overlay(images[src_idx], images[next(
        i for i, r in enumerate(roles) if r == "mask"
    )])
    kept = [merged if i == src_idx else im for i, im in enumerate(images)
            if roles[i] != "mask"]
    kept_roles = [roles[i] for i in range(len(images)) if roles[i] != "mask"]
    logger.info("独立 mask 已合并为高亮标注单图（源图索引=%d）", src_idx)
    return kept, kept_roles


# ============================================================
# 扩图（outpaint）画布扩展（任务 56）
# ============================================================

EXPAND_DIRECTIONS = ("left", "right", "up", "down", "all")

# 扩展区填充取样带宽度（px）——取"与扩展区相邻的那条边"的均值色，避免纯白填充的硬接缝
_EXPAND_BAND = 8


def align_up(value: int, align: int = SIZE_ALIGN) -> int:
    """向上取整到 align 的倍数（画布尺寸需 32 对齐，见 schemas.SIZE_ALIGN 说明）。"""
    return ((int(value) + align - 1) // align) * align


def expanded_size(width: int, height: int, direction: str, amount: int) -> tuple[int, int]:
    """计算扩图后的画布尺寸。

    先把手里的源图尺寸**向上对齐到 32**（非 32 对齐的源图会在右/下侧补最多 31px
    的填充色，保证画布 32 对齐、不被管线 `multiple_of=32` 静默取整），再按方向扩 amount。
    """
    if direction not in EXPAND_DIRECTIONS:
        raise ValueError(f"扩图方向非法：{direction}（可选：{'/'.join(EXPAND_DIRECTIONS)}）")
    amount = int(amount)
    if amount <= 0:
        raise ValueError(f"扩幅必须为正：{amount}")
    aw, ah = align_up(width), align_up(height)
    if direction == "all":
        return aw + 2 * amount, ah + 2 * amount
    if direction in ("left", "right"):
        return aw + amount, ah
    return aw, ah + amount


def _edge_mean_color(src: Image.Image, side: str, band: int = _EXPAND_BAND) -> tuple[int, ...]:
    """源图某条边的均值色（band px 宽度取样）。

    side：left / right / top / bottom；返回通道数与源图一致（RGBA 时 alpha 恒 255）。
    """
    arr = np.asarray(src, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"输入图像维度异常：{arr.shape}")
    h, w = arr.shape[0], arr.shape[1]
    b = max(1, min(band, h, w))
    if side == "left":
        seg = arr[:, :b]
    elif side == "right":
        seg = arr[:, w - b:]
    elif side == "top":
        seg = arr[:b, :]
    elif side == "bottom":
        seg = arr[h - b:, :]
    else:
        raise ValueError(f"边缘取样方向非法：{side}")
    mean = seg.reshape(-1, seg.shape[2]).mean(axis=0)
    rgb = [int(round(float(v))) for v in mean[:3]]
    if src.mode == "RGBA":
        return (rgb[0], rgb[1], rgb[2], 255)
    return (rgb[0], rgb[1], rgb[2])


def expand_canvas(
    src: Image.Image,
    direction: str,
    amount: int,
    fill: Optional[tuple[int, ...]] = None,
) -> Image.Image:
    """按方向扩展画布（扩图的核心预处理，任务 56）。

    做法：把源图**贴到偏移侧**（右扩→贴左、左扩→贴右、上扩→贴下、下扩→贴上、四周→居中），
    其余区域用 `fill` 填充；`fill` 缺省取「与扩展区相邻的那条边」的均值色
    （右扩取右边缘、左扩取左边缘、上扩取上边缘、下扩取下边缘、四周取四边均值）。

    为什么必须这样做（本轮实测）：仅给更大的输出尺寸而条件图仍是原图时，模型会**整体重绘**
    （原图区差异 69.8/255）；改成"扩展画布当输入"后原图区差异降到 6.7/255、接缝差 0.046。

    Args:
        src: 源图（PIL；RGB 或 RGBA —— RGBA 时填充 alpha=255）
        direction: left / right / up / down / all
        amount: 单边扩展像素（all 时四边各扩 amount）
        fill: 可选填充色（覆盖默认的边缘均值色）

    Returns:
        新的 PIL 图（尺寸 = expanded_size(...)）
    """
    new_w, new_h = expanded_size(src.width, src.height, direction, amount)
    w, h = src.width, src.height
    aw, ah = align_up(w), align_up(h)

    if fill is None:
        if direction == "all":
            edges = [_edge_mean_color(src, s) for s in ("left", "right", "top", "bottom")]
            fill = tuple(int(round(sum(c[i] for c in edges) / len(edges))) for i in range(len(edges[0])))
        else:
            adjacent = {"right": "right", "left": "left", "up": "top", "down": "bottom"}[direction]
            fill = _edge_mean_color(src, adjacent)

    # 1) 源图先补齐到 32 对齐（右/下各补最多 31px 的同一填充色）
    if aw != w or ah != h:
        base = Image.new(src.mode, (aw, ah), fill)
        base.paste(src, (0, 0))
    else:
        base = src

    # 2) 源图贴到"扩展方向的反侧"，让新区域留在被扩展的一侧
    if direction == "right":
        pos = (0, 0)
    elif direction == "left":
        pos = (new_w - aw, 0)
    elif direction == "down":
        pos = (0, 0)
    elif direction == "up":
        pos = (0, new_h - ah)
    else:  # all
        pos = (amount, amount)

    canvas = Image.new(src.mode, (new_w, new_h), fill)
    canvas.paste(base, pos)
    logger.info(
        "扩图：方向=%s 扩幅=%dpx → 画布 %d×%d（源图 %d×%d，对齐后 %d×%d 贴于 %s，填充色=%s）",
        direction, amount, new_w, new_h, w, h, aw, ah, pos, fill,
    )
    return canvas


# ============================================================
# 图片 staging
# ============================================================

def stage_images(image_inputs: list[Any], input_dir: Path) -> tuple[list[Image.Image], list[str]]:
    """把输入图统一落盘为 PIL（保持原顺序；**保留 RGBA**）。

    Args:
        image_inputs: ``ImageInput`` 对象或 dict（含 url / b64_json / path / role）
        input_dir: staging 目录（``config.input_dir``）

    Returns:
        (PIL 列表, role 列表) —— 顺序与入参一致（管线按此顺序编号「图1/图2…」）
    """
    out: list[Image.Image] = []
    roles: list[str] = []
    for i, item in enumerate(image_inputs):
        path = stage_one(item, input_dir, index=i)
        image = _open_validated(path)
        out.append(image)
        roles.append(_get(item, "role") or "unspecified")
    logger.info(
        "输入图 staging 完成：%d 张（roles=%s，sizes=%s）",
        len(out), describe_roles(roles), [im.size for im in out],
    )
    return out, roles


def stage_one(image_input: Any, input_dir: Path, index: int = 0) -> Path:
    """单张输入图落盘并返回路径。

    来源优先级（与既有 7007 引擎一致）：path → b64_json → url。
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    stem = f"input_{int(time.time())}_{uuid.uuid4().hex[:8]}_{index}"

    # 1) 本地路径（网关已下载 / 用户本地）
    raw_path = _get(image_input, "path")
    if raw_path:
        p = Path(raw_path)
        # 任务 60：本地路径白名单校验（防任意文件读取；网关归一化后一般已在允许根内）
        if p.is_absolute() and not _is_allowed_staging_path(p.resolve(), input_dir):
            raise ImageStagingError(
                f"输入图[{index}] 本地路径越界：{p.resolve()}（不允许读取）。"
                f"请先通过 POST /v1/files/image 上传，或使用 http(s) URL。"
            )
        if p.exists():
            return p
        logger.warning("ImageInput.path 不存在，尝试其它来源：%s", p)

    # 2) base64
    b64 = _get(image_input, "b64_json")
    if b64:
        try:
            raw = base64.b64decode(_strip_data_url(b64), validate=False)
        except Exception as e:  # noqa: BLE001
            raise ImageStagingError(f"输入图[{index}] base64 解码失败：{e}") from e
        if len(raw) > MAX_IMAGE_BYTES:
            raise ImageStagingError(
                f"输入图[{index}] 过大：{len(raw) / 1024**2:.1f}MB > {MAX_IMAGE_BYTES // 1024**2}MB"
            )
        p = input_dir / f"{stem}.png"
        p.write_bytes(raw)
        return p

    # 3) URL 下载
    url = _get(image_input, "url")
    if url:
        # 任务 60：非网络的绝对路径兜底（网关归一化后一般走 path 分支；
        # 此处覆盖「绕过网关直连引擎」时把本地路径塞进 url 字段的场景）
        lp = Path(url)
        if lp.is_absolute():
            resolved = lp.resolve()
            if not _is_allowed_staging_path(resolved, input_dir):
                raise ImageStagingError(
                    f"输入图[{index}] 本地路径越界：{resolved}（不允许读取）。"
                    f"请先通过 POST /v1/files/image 上传，或使用 http(s) URL。"
                )
            if resolved.exists():
                logger.info("输入图经 url 字段按本地路径读取：%s", resolved)
                return resolved
            raise ImageStagingError(f"输入图[{index}] 本地路径不存在：{resolved}")

        import httpx
        suffix = ".png"
        low = url.lower()
        if ".jpg" in low or ".jpeg" in low:
            suffix = ".jpg"
        elif ".webp" in low:
            suffix = ".webp"
        p = input_dir / f"{stem}{suffix}"
        try:
            with httpx.Client(timeout=httpx.Timeout(30, read=120), follow_redirects=True) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    total = 0
                    with open(p, "wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=1 << 20):
                            total += len(chunk)
                            if total > MAX_IMAGE_BYTES:
                                raise ImageStagingError(
                                    f"输入图[{index}] 下载超限：>{MAX_IMAGE_BYTES // 1024**2}MB"
                                )
                            fh.write(chunk)
        except ImageStagingError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ImageStagingError(f"输入图[{index}] 下载失败：{e}") from e
        logger.info("输入图已下载：%s -> %s", url[:80], p)
        return p

    raise ImageStagingError(f"输入图[{index}] 缺少 url/b64_json/path 来源字段")


# ============================================================
# 内部辅助
# ============================================================

def _allowed_staging_roots(input_dir: Path) -> list[Path]:
    """返回允许从本地磁盘读取参考图的根目录列表（任务 60 新增，防任意文件读取）。

    引擎与网关同机，远程调用方可借「参考图路径」读任意本地文件（``data/tasks.db``、
    系统文件），故必须白名单。允许根：
        - ``input_dir``（引擎 staging 临时区）
        - ``<data_root>/files``（网关上传缓存：input_image / input_audio / input_video）
        - ``<project_root>/output``（引擎产物，允许二次编辑引用）

    由 ``input_dir``（``<project_root>/data/inputs/<engine_id>``）反推 data_root / project_root。
    """
    roots: list[Path] = [input_dir]
    try:
        data_root = input_dir.parents[1]      # data/inputs/<engine_id> → data
        roots.append(data_root / "files")
        roots.append(data_root.parent / "output")
    except IndexError:  # pragma: no cover - 非常规布局时退化为仅 input_dir
        pass
    return roots


def _is_allowed_staging_path(p: Path, input_dir: Path) -> bool:
    """判断本地路径是否落在允许根内（``resolve()`` 后比较，防符号链接绕过）。"""
    for root in _allowed_staging_roots(input_dir):
        try:
            if p.is_relative_to(root.resolve()):
                return True
        except (OSError, ValueError):
            continue
    return False


def _get(obj: Any, key: str) -> Any:
    """同时支持 pydantic 对象与 dict 取值。"""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _strip_data_url(b64: str) -> str:
    """去掉 data URL 前缀（``data:image/png;base64,``）。"""
    if b64.startswith("data:"):
        try:
            return b64.split(",", 1)[1]
        except IndexError:
            return b64
    return b64


def _open_validated(path: Path) -> Image.Image:
    """打开并校验尺寸（保留原色彩模式——RGBA 不丢 alpha）。"""
    try:
        image = Image.open(path)
        image.load()
    except Exception as e:  # noqa: BLE001
        raise ImageStagingError(f"输入图无法解码：{path.name}（{e}）") from e
    w, h = image.size
    long_side, short_side = max(w, h), min(w, h)
    if short_side < MIN_SIDE:
        raise ImageStagingError(
            f"输入图过小：{path.name} {w}×{h}（最短边 {short_side} < {MIN_SIDE}）。"
            f"素材清晰度直接影响编辑稳定性，请换更清晰的图"
        )
    if long_side > MAX_SIDE:
        raise ImageStagingError(f"输入图过大：{path.name} {w}×{h}（最长边 > {MAX_SIDE}）")
    return image
