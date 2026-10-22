"""load() 只校验不下载：文件完整性校验 + 缺失信息指引（项目规则 §2）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from engines.qwen_image_21_feeb8130 import pipeline_builder as pb
from engines.qwen_image_21_feeb8130.pipeline_builder import (
    MIN_SIZES,
    SHARED_FILES,
    ModelFilesError,
    check_plan_files,
    check_shared_files,
)


def test_shared_files_list_refs_valid_min_keys():
    for rel, min_key in SHARED_FILES:
        if min_key:
            assert min_key in MIN_SIZES, f"{rel} 引用未定义的 MIN_SIZES 键 {min_key}"


def test_shared_files_use_processor_subdir():
    """2.1 仓 tokenizer 文件在 processor/ 下（本轮修复点），不得引用 tokenizer/。"""
    for rel, _ in SHARED_FILES:
        assert not rel.startswith("tokenizer/"), f"{rel} 指向不存在的 tokenizer/ 目录"


def test_check_shared_files_missing_raises_with_guidance(tmp_path):
    with pytest.raises(ModelFilesError) as ei:
        check_shared_files(tmp_path)
    msg = str(ei.value)
    assert "模型管理" in msg
    assert "下载" in msg
    assert "本地路径" in msg


def test_check_shared_files_detects_truncated_file(tmp_path):
    """残文件（存在但大小不足）必须被判为不完整。"""
    # 造一个存在但很小的分片（阈值 4.4GB）
    p = tmp_path / "text_encoder" / "model-00001-of-00004.safetensors"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"0" * 1024)
    with pytest.raises(ModelFilesError) as ei:
        check_shared_files(tmp_path)
    assert "model-00001-of-00004.safetensors" in str(ei.value)


def test_check_plan_files_missing_raises(tmp_path):
    class _Plan:
        scheme = type("S", (), {"id": "t2_gguf_q8"})()
        variant = "qwen-image-21-official-gguf-q8"
        dit_path = tmp_path / "nonexistent.gguf"

    with pytest.raises(ModelFilesError):
        check_plan_files(_Plan(), tmp_path)


def test_check_plan_files_small_safetensors_raises(tmp_path):
    f = tmp_path / "qwen-image-2.1-Q8_0.gguf"
    f.write_bytes(b"0" * (1024 * 1024))  # 1MB < 1GB 阈值

    class _Plan:
        scheme = type("S", (), {"id": "t2_gguf_q8"})()
        variant = "qwen-image-21-official-gguf-q8"
    _Plan.dit_path = f

    with pytest.raises(ModelFilesError) as ei:
        check_plan_files(_Plan(), tmp_path)
    assert "大小不足" in str(ei.value)


def test_check_plan_files_ok_for_non_weight_suffix(tmp_path):
    """体积阈值只对 .gguf/.safetensors 生效，其它后缀仅校验存在性（避免测试写 GB 级文件）。"""
    f = tmp_path / "dummy.bin"
    f.write_bytes(b"x")

    class _Plan:
        scheme = type("S", (), {"id": "t2_gguf_q8"})()
        variant = "qwen-image-21-official-gguf-q8"
    _Plan.dit_path = f

    check_plan_files(_Plan(), tmp_path)  # 不抛


def test_no_network_download_in_load_path():
    """load() 路径禁止 web 下载：pipeline_builder 源码不得出现 snapshot_download/hf_hub_download。"""
    src = Path(pb.__file__).read_text(encoding="utf-8")
    for forbidden in ("snapshot_download", "hf_hub_download", "requests.get", "urllib.request"):
        assert forbidden not in src, f"pipeline_builder 出现网络下载调用：{forbidden}"


def test_gguf_backend_no_download_but_offline_only():
    """GGUF 后端同样不得联网下载（离线加载走内置 config）。"""
    from engines.qwen_image_21_feeb8130.backends import gguf_backend

    src = Path(gguf_backend.__file__).read_text(encoding="utf-8")
    for forbidden in ("snapshot_download", "hf_hub_download"):
        assert forbidden not in src, f"gguf_backend 出现网络下载调用：{forbidden}"
