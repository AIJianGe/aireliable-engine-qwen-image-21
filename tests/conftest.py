"""pytest 公共夹具（任务 48：Qwen-Image2.1 引擎单元测试）。

- 把项目根加入 sys.path，使 ``engines.qwen_image_21_feeb8130.*`` 可导入
- 提供 ENGINE_DIR / ROOT 常量
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ENGINE_DIR = Path(__file__).resolve().parents[1]
ROOT = ENGINE_DIR.parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def engine_dir() -> Path:
    return ENGINE_DIR


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT
