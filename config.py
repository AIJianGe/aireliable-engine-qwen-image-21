"""Qwen-Image-2.1 图像引擎配置（engine_id: qwen_image_21_feeb8130）。

从环境变量 + 命令行参数加载，提供引擎运行所需的全部配置。

与 7007 引擎（qwen_image_5e8f3a21）一致的约定：
    - 模型目录优先读环境变量 ``AIRELIABLE_MODEL_LAYOUT_BASE``（由平台 EngineManager
      启动子进程时同步 manifest.models.model_layout.base）
    - 设备策略 ``AIRELIABLE_DEVICE_POLICY``：auto / gpu / gpu+cpu / cpu
    - 新增：默认精度档 ``default_scheme``（4 档 T1–T4 之一）
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

from aireliable_engine import EngineConfig as BaseEngineConfig

# ===== 常量 =====

ENGINE_ID = "qwen_image_21_feeb8130"
ENGINE_NAME = "Qwen-Image2.1"
CAPABILITY = "text-to-image"
ENGINE_TYPE = "image"

# 默认端口（与 config/engines/qwen_image_21_feeb8130.yaml 一致；7001-7014 已占用，7009 空闲）
DEFAULT_PORT = 7009

# 模型目录布局（相对 MODELS_ROOT）；优先读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE
_DEFAULT_MODEL_SUBDIR = "image/qwen-image-2.1/"

# 默认精度档（4 档之一，见 schemes/）
DEFAULT_SCHEME = "t2_gguf_q8"

# 设备策略合法取值（与 7007 引擎一致）
DEVICE_POLICIES = ("auto", "gpu", "gpu+cpu", "cpu")


def _resolve_model_subdir() -> str:
    """解析模型子目录（优先读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE）。"""
    env_val = os.environ.get("AIRELIABLE_MODEL_LAYOUT_BASE")
    if env_val:
        return env_val.rstrip("/") + "/"
    return _DEFAULT_MODEL_SUBDIR


MODEL_SUBDIR = _resolve_model_subdir()


@dataclass
class EngineConfig(BaseEngineConfig):
    """Qwen-Image-2.1 图像引擎运行配置。"""

    # ===== 基础（身份 / 端口默认值由本引擎覆盖，其余公共字段见基类）=====
    engine_id: str = ENGINE_ID
    engine_name: str = ENGINE_NAME
    capability: str = CAPABILITY
    engine_type: str = ENGINE_TYPE
    port: int = DEFAULT_PORT

    # ===== 推导路径 =====
    model_dir: Path = field(init=False)
    lora_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    input_dir: Path = field(init=False)
    schemes_dir: Path = field(init=False)

    # ===== 精度档位（4 档 T1–T4）=====
    # 用户/UI 显式指定的运行精度；auto = 由 DtypeResolver 按显卡自适应
    precision: str = "auto"
    # 默认精度档（未显式指定 variant/scheme 时使用）
    default_scheme: str = DEFAULT_SCHEME

    def __post_init__(self) -> None:
        """推导子路径。"""
        self.model_dir = self.models_root / MODEL_SUBDIR

        lora_env = os.environ.get("AIRELIABLE_LORA_DIR")
        if lora_env:
            p = Path(lora_env)
            self.lora_dir = p if p.is_absolute() else self.models_root / p
        else:
            self.lora_dir = self.model_dir / "loras"

        self.log_dir = self.project_root / "logs" / "engines"
        self.input_dir = self.data_root / "inputs" / self.engine_id
        self.schemes_dir = Path(__file__).parent / "schemes"

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        """解析推理设备（device="auto" 在 load() 入口统一解析）。"""
        if self.device_policy == "cpu":
            return "cpu"
        if self.device == "cpu":
            return "cpu"
        if self.device.startswith("cuda"):
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def to_dict(self) -> dict:
        """转 dict（用于日志/调试）。"""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "capability": self.capability,
            "engine_type": self.engine_type,
            "port": self.port,
            "host": self.host,
            "models_root": str(self.models_root),
            "model_dir": str(self.model_dir),
            "lora_dir": str(self.lora_dir),
            "output_root": str(self.output_root),
            "log_dir": str(self.log_dir),
            "input_dir": str(self.input_dir),
            "schemes_dir": str(self.schemes_dir),
            "device": self.device,
            "device_policy": self.device_policy,
            "precision": self.precision,
            "default_scheme": self.default_scheme,
            "log_level": self.log_level,
        }


def parse_args() -> EngineConfig:
    """解析命令行参数 + 环境变量，构造 EngineConfig。

    优先级：命令行 > 环境变量 > 默认值
    """
    parser = argparse.ArgumentParser(description="Qwen-Image-2.1 图像生成引擎（t2i，4 档精度）")

    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP 端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="绑定地址（默认 127.0.0.1）")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备：auto / cpu / cuda:0（默认 auto）")
    parser.add_argument("--device-policy", type=str, default=None,
                        help="设备策略：auto / gpu / gpu+cpu / cpu（默认 auto；"
                             "未传时读环境变量 AIRELIABLE_DEVICE_POLICY）")
    parser.add_argument("--precision", type=str, default=None,
                        help="运行精度：auto / bf16 / fp16 / fp32 / int8（默认 auto）")
    parser.add_argument("--scheme", type=str, default=None,
                        help=f"默认精度档：t1_bf16 / t2_gguf_q8 / t3_gguf_q4 / t4_int8"
                             f"（默认 {DEFAULT_SCHEME}）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别（默认 INFO）")
    parser.add_argument("--models-root", type=str, default=None,
                        help="模型根目录（默认 models/）")

    args = parser.parse_args()

    models_root = Path(args.models_root or os.environ.get("AIRELIABLE_MODELS_ROOT", "models"))
    project_root = Path(os.environ.get("AIRELIABLE_PROJECT_ROOT", ".")).resolve()
    output_root_env = os.environ.get("AIRELIABLE_OUTPUT_ROOT")
    output_root = Path(output_root_env) if output_root_env else project_root / "output"
    data_root = Path(os.environ.get("AIRELIABLE_DATA_ROOT", project_root / "data"))

    device_policy = (
        args.device_policy
        or os.environ.get("AIRELIABLE_DEVICE_POLICY")
        or "auto"
    ).strip()
    if device_policy not in DEVICE_POLICIES:
        raise ValueError(
            f"非法设备策略：{device_policy!r}（合法值：{DEVICE_POLICIES}）。"
        )

    precision = (
        args.precision
        or os.environ.get("AIRELIABLE_PRECISION")
        or "auto"
    ).strip()
    scheme = (
        args.scheme
        or os.environ.get("AIRELIABLE_SCHEME")
        or DEFAULT_SCHEME
    ).strip()

    return EngineConfig(
        port=args.port,
        host=args.host,
        device=args.device,
        device_policy=device_policy,
        precision=precision,
        default_scheme=scheme,
        log_level=args.log_level,
        models_root=models_root,
        project_root=project_root,
        output_root=output_root,
        data_root=data_root,
    )
