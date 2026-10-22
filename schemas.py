"""Qwen-Image-2.1 引擎输入/输出 schema（t2i / i2i / edit 三模式）。

请求输入（ImageInputs，经 InferRequest.inputs 传入）：
    - prompt / negative_prompt
    - mode：t2i（文生图）/ i2i（图生图多图参考）/ edit（图像编辑）
      None = 由输入图张数推断（0→t2i，1→edit，≥2→i2i）；``request.workflow`` 可作显式来源
    - width / height（**32 对齐**，256–2048；**None = 由管线按条件图推算**）
      ⚠️ 32 对齐是**管线硬约束**（任务 56 源码核实）：`QwenImage21Pipeline.__call__` 内
      `multiple_of = vae_scale_factor * 2 = 32`，非 32 对齐的宽高会被**静默向下取整**
      （1360→1344）→ "页面显示 ≠ 实际输出"。故此处**显式拒绝**（给可读报错），
      而不是让它悄悄变值。
    - expand：扩图（outpaint）参数（任务 56 新增，见 ExpandSpec）；对 **edit 单源图**生效，
      生效时**以扩展画布尺寸为输出尺寸**（忽略显式 width/height）
    - num_inference_steps / true_cfg_scale / seed
    - precision：运行精度策略（auto / bf16 / fp16 / fp32 / int8）
    - scheme：精度档位（t1_bf16 / t2_gguf_q8 / t3_gguf_q4 / t4_int8）

图片输入（**两条等价通道**，任务 49 分析文档 §4.1）：
    1. ``InferRequest.images``（网关链路：``ImageInput{url|b64_json|path, role}``）
    2. ``ImageInputs.images``（本字段；dict 形式，供引擎自带 WebUI 直连 / extra.inputs 透传）

    role 语义（SDK）：source（源图）/ mask（独立蒙版）/ reference（参考图）；
    Qwen-Image-2.1 局部编辑 = 「图上圈选/涂抹」或「原图 + 独立 mask 两张输入」。

⚠️ denoise：Qwen-Image-2.1 **无 SD 式 img2img 语义**（``image=`` 是条件图而非加噪起点）
    —— 本字段仅登记，引擎忽略并记 WARNING（分析文档 §3.2 / §6.1 方案 A）。
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# 推理模式（一期）：文生图 / 图生图多图参考 / 图像编辑
Mode = Literal["t2i", "i2i", "edit"]

# 运行精度策略
Precision = Literal["auto", "bf16", "fp16", "fp32", "int8"]

# 精度档位（4 档 T1–T4）
Scheme = Literal["t1_bf16", "t2_gguf_q8", "t3_gguf_q4", "t4_int8"]

# ===== 尺寸契约（任务 56）=====
SIZE_ALIGN = 32        # 管线 multiple_of = vae_scale_factor(16) × 2
SIZE_MIN = 256
SIZE_MAX = 2048

# 扩图方向
ExpandDirection = Literal["left", "right", "up", "down", "all"]
EXPAND_DIRECTION_LABELS: dict[str, str] = {
    "left": "左", "right": "右", "up": "上", "down": "下", "all": "四周",
}
EXPAND_AMOUNT_MAX = 1024   # 单边最大扩幅（画布受 SIZE_MAX 约束，引擎侧再校验成品画布）

# 扩图自动提示词前缀（保守指令 —— 实测缺失它会被"改写画面"而非"扩图"）
EXPAND_PROMPT_TEMPLATE = (
    "保持原有主体、材质细节与背景完全不变，将画面向{dir}扩展并补全空白区域，"
    "与原有内容自然无缝衔接，不要添加任何新物体。{prompt}"
)

# 透明图官方推荐 prompt 模板（分析文档 §2.1）
TRANSPARENT_PROMPT_TEMPLATE = (
    "This is an RGBA image with transparency. {prompt}. "
    "The image has alpha channel and the background is transparent."
)


class ExpandSpec(BaseModel):
    """扩图（outpaint）参数（任务 56 新增）。

    语义：以**源图**为基准把画布按方向做大，待补区域用「与扩展区相邻的源图边缘均值色」
    填充，再把**扩展后的画布当条件图**喂管线，并自动注入保守提示词前缀。

    为什么必须"扩展画布当输入"（本轮实测）：
        - 仅改输出尺寸（条件图仍是原图）→ 模型**整体重绘**：原图区差异 69.8/255
        - 画布扩展输入 + 保守指令 → 原图区差异 **6.7/255**、接缝差 **0.046**（干净扩图）
    """

    direction: ExpandDirection = "right"
    """扩展方向：left / right / up / down / all（all = 四周各扩 amount）。"""

    amount: int = 256
    """单边扩展像素（32 对齐，32–1024）。输出画布尺寸：
    right / left → (W+amount) × H；up / down → W × (H+amount)；all → (W+2a) × (H+2a)。"""

    auto_prompt: bool = True
    """是否自动注入保守提示词前缀（EXPAND_PROMPT_TEMPLATE）。"""

    @field_validator("amount")
    @classmethod
    def _check_amount(cls, v: int) -> int:
        if v < SIZE_ALIGN or v > EXPAND_AMOUNT_MAX:
            raise ValueError(f"扩幅越界（{SIZE_ALIGN}-{EXPAND_AMOUNT_MAX}）：{v}")
        if v % SIZE_ALIGN != 0:
            raise ValueError(f"扩幅须 {SIZE_ALIGN} 对齐（画布尺寸需 {SIZE_ALIGN} 对齐）：{v}")
        return v


class ImageInputs(BaseModel):
    """推理输入参数（/infer 的 inputs 字段）。"""

    prompt: str = ""
    negative_prompt: Optional[str] = None

    mode: Optional[Mode] = None
    """推理模式；None = 由图片张数推断（见模块 docstring）。"""

    width: Optional[int] = None
    height: Optional[int] = None
    """None = 交给管线按条件图宽高比推算；
    显式给出时做 **32 对齐** + 范围（256–2048）校验（管线 multiple_of=32，非 32 会静默取整）。"""

    expand: Optional[ExpandSpec] = None
    """扩图参数（任务 56）：仅对 edit 单源图生效；生效时输出尺寸 = 扩展画布尺寸，
    显式 width/height 被忽略（引擎记日志说明）。"""

    num_inference_steps: Optional[int] = None
    """去噪步数；None = 按 variant 契约（Turbo→8，其余→40）。"""

    true_cfg_scale: float = 1.0
    """真 CFG 强度（Qwen-Image-2.1 发布路径默认 1.0，即跳过负向引导；
    QwenImage21Pipeline 无独立 guidance_scale 参数——已实测确认）"""

    seed: Optional[int] = None

    precision: Optional[Precision] = None
    scheme: Optional[Scheme] = None

    # ===== 任务 49 新增：三模式扩展字段 =====
    images: Optional[list[dict[str, Any]]] = None
    """图片输入（dict 形式：{url|b64_json|path, role}）。
    与 ``InferRequest.images`` 等价；网关链路走后者，本字段供直连 WebUI / extra.inputs 使用。"""

    transparent: bool = False
    """透明背景开关：True 时自动套官方 RGBA prompt 模板（输出 alpha 通道）。"""

    denoise: Optional[float] = None
    """登记字段：Qwen-Image-2.1 无 SD 式 img2img 语义 → 引擎忽略并告警。"""

    max_sequence_length: int = 512
    """VL 编码长度（上限 1024）"""

    @field_validator("num_inference_steps")
    @classmethod
    def _check_steps(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 100):
            raise ValueError(f"num_inference_steps 越界（1-100）：{v}")
        return v

    @field_validator("true_cfg_scale")
    @classmethod
    def _check_cfg(cls, v: float) -> float:
        if v < 0 or v > 20:
            raise ValueError(f"CFG 越界（0-20）：{v}")
        return v

    @field_validator("width", "height")
    @classmethod
    def _check_size(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v  # 留空：交给管线按条件图推算
        if v < SIZE_MIN or v > SIZE_MAX:
            raise ValueError(f"宽高越界（{SIZE_MIN}-{SIZE_MAX}）：{v}")
        if v % SIZE_ALIGN != 0:
            raise ValueError(
                f"宽高须 {SIZE_ALIGN} 对齐：{v}"
                f"（管线 multiple_of={SIZE_ALIGN}，非对齐会被静默向下取整 → 输出尺寸与请求不符）"
            )
        return v

    @field_validator("max_sequence_length")
    @classmethod
    def _check_seq_len(cls, v: int) -> int:
        if v < 64 or v > 1024:
            raise ValueError(f"max_sequence_length 越界（64-1024）：{v}")
        return v

    @field_validator("denoise")
    @classmethod
    def _check_denoise(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 1):
            raise ValueError(f"denoise 越界（0-1）：{v}")
        return v


class HistoryItem(BaseModel):
    """生成历史条目（/history 端点返回）。"""

    task_id: str
    mode: Mode = "t2i"
    prompt: str = ""
    image_url: str = ""
    thumbnail_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    precision: Optional[str] = None
    loras: list[str] = Field(default_factory=list)
    created_at: float = 0.0
    duration_seconds: float = 0.0
