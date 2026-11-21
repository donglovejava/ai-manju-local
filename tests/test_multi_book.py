"""多书支持与分镜拆分修复的回归测试。

覆盖本轮修复的三个真实 bug：
  1. 超长场景无法被 chunk_scenes 切开 → 分镜 JSON 截断
  2. 角色卡目录/角色名硬编码 → 无法支持第二本小说
  3. 近景缺少「单人物」约束 → 出图崩成群像拼贴
"""
from pathlib import Path

import yaml

from prompt_builder import SHOT_TYPES, PromptBuilder
from storyboard import chunk_scenes, split_scene

ROOT = Path(__file__).resolve().parent.parent


# ---------- 1. 超长场景拆分 ----------

def test_split_scene_splits_by_lines():
    sc = {"scene_id": "c001_s01", "heading": "长场景", "lines": ["一二三四五"] * 10}
    parts = split_scene(sc, max_chars=25)
    assert len(parts) > 1
    for p in parts:
        assert p["chars"] <= 25
        assert p["heading"] == "长场景"
        assert p["scene_id"] == "c001_s01"
    # 所有行都要保留，不能丢内容
    total = sum(len(p["lines"]) for p in parts)
    assert total == 10


def test_split_scene_keeps_short_scene_intact():
    sc = {"scene_id": "s", "heading": "短", "lines": ["abc", "def"]}
    parts = split_scene(sc, max_chars=100)
    assert len(parts) == 1
    assert parts[0]["lines"] == ["abc", "def"]


def test_chunk_scenes_handles_oversized_single_scene():
    """一个 2000 字的场景，必须被拆成多个 ≤600 字的块。"""
    long_lines = ["这是一句大约二十个字的叙述内容用于测试。" * 1] * 100
    chapter = {
        "index": 1,
        "scenes": [
            {"scene_id": "s1", "heading": "甲", "lines": ["短句"], "chars": 2},
            {"scene_id": "s2", "heading": "乙", "lines": long_lines, "chars": 2000},
        ],
    }
    chunks = chunk_scenes(chapter, max_chars=600)
    assert len(chunks) >= 3
    for ch in chunks:
        size = sum(len(l) for sc in ch["scenes"] for l in sc["lines"])
        assert size <= 600, f"块过大: {size}"


def test_chunk_scenes_preserves_all_text():
    chapter = {
        "index": 1,
        "scenes": [{"scene_id": "s", "heading": "h", "lines": [f"line{i}" for i in range(50)], "chars": 300}],
    }
    chunks = chunk_scenes(chapter, max_chars=100)
    got = [line for ch in chunks for sc in ch["scenes"] for line in sc["lines"]]
    assert got == [f"line{i}" for i in range(50)]


# ---------- 2. 多书支持 ----------

def test_prompt_builder_reads_chars_dir_from_config():
    cfg = yaml.safe_load((ROOT / "config_wzn.yaml").read_text(encoding="utf-8"))
    pb = PromptBuilder(cfg)
    assert set(pb.chars) == {"沈迟", "苏锦"}, "第二本书的角色卡未按 config.paths.chars 加载"


def test_default_config_still_loads_original_chars():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pb = PromptBuilder(cfg)
    assert set(pb.chars) == {"陈屿", "林晚"}


def test_wzn_config_declares_characters():
    cfg = yaml.safe_load((ROOT / "config_wzn.yaml").read_text(encoding="utf-8"))
    assert cfg["characters"] == ["沈迟", "苏锦"]
    assert "我去找你" in cfg["novel"]["path"]
    assert cfg["paths"]["work"].endswith("work_wzn")
    assert cfg["paths"]["output"].endswith("output_wzn")


def test_wzn_character_cards_have_required_fields():
    from prompt_builder import CharacterCard

    for name in ("沈迟", "苏锦"):
        card = CharacterCard.load(ROOT / "assets" / "chars_wzn" / f"{name}.yaml")
        assert card.name == name
        assert card.age_lock, f"{name} 缺少 age_lock"
        assert card.default_outfit
        assert card.outfits


# ---------- 3. 近景单人物约束 ----------

def test_close_shots_enforce_single_person():
    """出图崩坏主因是近景脑补群像（拼贴/双人）。

    2026-11 实测：Illustrious 对自然语言 "one person only" 几乎无反应，
    真正起作用的是 danbooru 的 solo / solo focus 标签。加标签后拼贴消失，
    但 LoRA 强度必须锁在 0.65 —— 0.55 会退化成女性，0.72 也会翻车。
    """
    assert "solo" in SHOT_TYPES["close"]
    assert "solo focus" in SHOT_TYPES["close"]
    assert "one face only" in SHOT_TYPES["close"]
    assert "solo" in SHOT_TYPES["extreme_close"]
    assert "one single detail" in SHOT_TYPES["extreme_close"]


def test_every_shot_type_is_non_empty_english():
    for key, value in SHOT_TYPES.items():
        assert value.strip(), f"{key} 为空"
        assert not any("\u4e00" <= ch <= "\u9fff" for ch in value), f"{key} 含中文，SDXL 不认"


# ---------- 4. 按景别动态调 LoRA 强度 ----------

def test_shot_lora_strength_by_shot_type():
    from run_episode import shot_lora_strength

    img_cfg = {
        "lora_strength_by_shot": {"close": 0.62, "wide": 0.85},
        "lora_strength_default": 0.7,
    }
    assert shot_lora_strength(img_cfg, {"shot_type": "close"}) == 0.62
    assert shot_lora_strength(img_cfg, {"shot_type": "wide"}) == 0.85
    # 未登记的景别走默认值
    assert shot_lora_strength(img_cfg, {"shot_type": "unknown"}) == 0.7
    # 缺字段也走默认
    assert shot_lora_strength(img_cfg, {}) == 0.7
    # 命令行覆盖优先
    assert shot_lora_strength(img_cfg, {"shot_type": "close"}, override=0.9) == 0.9


def test_wzn_config_has_lora_strength_table():
    cfg = yaml.safe_load((ROOT / "config_wzn.yaml").read_text(encoding="utf-8"))
    table = cfg["image"]["lora_strength_by_shot"]
    # 近景必须低于全景（近景压低防幼态，全景拉高锁形象）
    assert table["close"] < table["wide"]
    assert table["extreme_close"] < table["medium"] < table["wide"]
