"""manifest.yaml 一致性（项目规则：model_files 覆盖 load() 全部依赖 + variant 三处同步）。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from engines.qwen_image_21_feeb8130 import layout
from engines.qwen_image_21_feeb8130.config import ENGINE_ID
from engines.qwen_image_21_feeb8130.model_scanner import VARIANT_META

ENGINE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load((ENGINE_DIR / "manifest.yaml").read_text(encoding="utf-8"))


def test_engine_identity(manifest):
    assert manifest["engine_id"] == ENGINE_ID
    assert manifest["paradigm"] == "integrated"
    assert manifest["export_mode"] == "source"


def test_model_layout_base_matches_layout(manifest):
    base = manifest["models"]["model_layout"]["base"]
    assert base == layout.MODEL_BASE, f"manifest base={base!r} 与 layout.MODEL_BASE 不一致"


def test_model_file_ids_unique_and_nonempty(manifest):
    ids = [f["id"] for f in manifest["models"]["model_files"]]
    assert len(ids) == len(set(ids)), "model_files 存在重复 id"
    assert len(ids) > 0


def test_every_model_file_has_source_urls(manifest):
    """每个 model_file 必须声明 source.url + source.mirror_url（规则 §1 反面案例②）。"""
    for f in manifest["models"]["model_files"]:
        src = f.get("source") or {}
        assert src.get("url"), f"{f['id']} 缺 source.url"
        assert src.get("mirror_url"), f"{f['id']} 缺 source.mirror_url"
        assert f.get("filename"), f"{f['id']} 缺 filename"
        assert f.get("target_dir"), f"{f['id']} 缺 target_dir"


def test_variant_file_ids_subset_of_declared(manifest):
    declared = {f["id"] for f in manifest["models"]["model_files"]}
    for v in manifest["models"]["variants"]:
        unknown = set(v["model_file_ids"]) - declared
        assert not unknown, f"variant {v['name']} 引用未声明文件：{unknown}"


def test_variant_union_covers_all_files(manifest):
    declared = {f["id"] for f in manifest["models"]["model_files"]}
    used: set[str] = set()
    for v in manifest["models"]["variants"]:
        used |= set(v["model_file_ids"])
    assert used == declared, f"未覆盖的 model_files：{declared - used}（浪费/漏声明）"


def test_dit_files_mutually_exclusive(manifest):
    """6 个 DiT 权重必须标 exclusive_group，且同组内互斥（防一键下 61.78GB）。"""
    dits = [f for f in manifest["models"]["model_files"] if f.get("role") == "main"]
    assert len(dits) == 6, f"预期 6 个 DiT，实际 {len(dits)}"
    for f in dits:
        assert f.get("exclusive_group") == "transformer", f"{f['id']} 未标 exclusive_group"
    for v in manifest["models"]["variants"]:
        variant_dits = [i for i in v["model_file_ids"] if i.startswith("qi21-dit-")]
        assert len(variant_dits) == 1, f"variant {v['name']} 声明的 DiT 数 {len(variant_dits)}≠1（互斥被破坏）"


def test_shared_files_not_exclusive(manifest):
    """共享组件（编码器/VAE/tokenizer/scheduler）不得标 exclusive_group。"""
    for f in manifest["models"]["model_files"]:
        if f.get("role") in ("text_encoder", "vae", "tokenizer", "scheduler"):
            assert "exclusive_group" not in f, f"共享文件 {f['id']} 不应有 exclusive_group"


def test_variant_names_synced_three_places(manifest):
    """三处同步：manifest variants / layout.VARIANT_DIT / model_scanner.VARIANT_META。"""
    mv = {v["name"] for v in manifest["models"]["variants"]}
    assert mv == set(layout.VARIANT_DIT), f"manifest↔VARIANT_DIT 不一致：{mv ^ set(layout.VARIANT_DIT)}"
    assert mv == set(VARIANT_META), f"manifest↔VARIANT_META 不一致：{mv ^ set(VARIANT_META)}"


def test_default_variant_valid(manifest):
    dv = manifest["models"]["default_variant"]
    names = {v["name"] for v in manifest["models"]["variants"]}
    assert dv in names, f"default_variant={dv!r} 不在 variants 中"


def test_variant_path_matches_layout_dit(manifest):
    """variant.path 的文件名必须与 layout.VARIANT_DIT 一致。"""
    for v in manifest["models"]["variants"]:
        expected = layout.VARIANT_DIT[v["name"]]
        assert v["path"].endswith(expected), f"{v['name']}: path={v['path']!r} 与 {expected!r} 不一致"


def test_layout_tokenizer_subdir_is_processor():
    """Qwen-Image-2.1 仓无 tokenizer/ 目录，文件在 processor/ 下（本轮修复点之一）。"""
    assert layout.TOKENIZER_SUBDIR == "processor"


def test_offline_config_dir_is_correct_transformer_config():
    """引擎内置 transformer config（from_single_file 离线用）必须与官方架构一致。"""
    import json

    cfg_path = layout.OFFLINE_CONFIG_DIR / "config.json"
    assert cfg_path.exists(), "内置 config.json 不存在（离线 from_single_file 会失败）"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["_class_name"] == "QwenImage21Transformer2DModel"
    assert cfg["context_in_dim"] == 4096
    assert cfg["num_attention_heads"] * cfg["attention_head_dim"] == 4096
    assert cfg["num_layers"] == 32
