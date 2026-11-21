"""分镜 → SDXL 提示词。

三块拼装：
  1. 风格锁（config.style.positive_prefix / negative）
  2. 角色卡（assets/chars/*.yaml）——保证同一角色跨镜头描述完全一致
  3. 镜头描述（景别 + 动作 + 场景 + 光线）

角色卡的字段会被固定顺序拼接，任何镜头都不允许模型自由发挥外貌词，
这是 8GB 本地出图能维持角色一致性的第一道保险（第二道是 IP-Adapter）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# 景别 → 英文镜头词。竖屏漫剧以中近景为主，全景只用于交代环境。
# 近景必须显式写「只有一个人」，否则模型极易脑补成群像拼贴（实测崩坏主因）。
SHOT_TYPES = {
    "extreme_wide": "extreme wide shot, establishing shot, environment dominant, characters small in frame",
    "wide": "wide shot, full body, characters visible head to toe",
    "medium": "medium shot, cowboy shot, upper body, single subject",
    # solo / solo focus 是 danbooru 单人强标签；自然语言 "one person only" 对 Illustrious 几乎无效
    "close": "close-up, head and shoulders, solo, solo focus, single head, one face only",
    "extreme_close": (
        "extreme close-up, solo, solo focus, one single detail, "
        "one face or one hand only, tight framing, shallow depth of field"
    ),
    "over_shoulder": "over-the-shoulder shot, foreground shoulder blurred, single subject",
    "pov": "first person pov, hands visible, subjective camera",
}

# 情绪 → 光线/氛围词
MOODS = {
    "平静": "soft even lighting, calm atmosphere",
    "压抑": "low key lighting, heavy shadow, muted desaturated colors",
    "紧张": "hard directional light, high contrast, tense atmosphere",
    "温柔": "warm golden light, soft focus, gentle atmosphere",
    "悲伤": "cool blue tones, dim light, melancholic atmosphere",
    "心动": "warm soft glow, slight bloom, romantic atmosphere",
    "尴尬": "flat cold fluorescent light, awkward atmosphere",
    "怀旧": "sepia toned, faded film grain, nostalgic atmosphere",
    "夜晚": "night scene, street lamp glow, deep blue shadows",
    "雨天": "rain, wet reflective surfaces, overcast light",
    "雪": "snow, cold white light, breath vapor",
}

# 构图补充：竖屏 9:16 的安全区意识
COMPOSITION_HINTS = {
    "close": "rule of thirds, eyes on upper third",
    "extreme_close": "centered detail, negative space on top",
    "medium": "character centered slightly low, headroom for subtitles",
    "wide": "character on lower third, large sky or ceiling negative space",
}


GENDER_TAG = {"male": "1boy", "man": "1boy", "female": "1girl", "woman": "1girl"}


@dataclass
class CharacterCard:
    name: str
    en_name: str
    gender: str
    age: int
    hair: str
    face: str
    build: str
    default_outfit: str
    outfits: dict[str, str]
    age_lock: str = ""
    notes: str = ""

    def describe(self, outfit: str | None = None) -> str:
        cloth = self.outfits.get(outfit or "", "") or self.default_outfit
        # age_lock 是高权重短语，放在最前面，压制模型把 30 岁画成 17 岁的倾向
        age_part = self.age_lock or f"{self.age} years old"
        # 1boy/1girl 是 Illustrious 最强的性别信号。只靠句子里顺带提到 male，
        # 模型经常画成女性（实测 043/045 两张就栽在这里）。
        tag = GENDER_TAG.get((self.gender or "").lower(), "")
        parts = [tag, age_part, self.gender, self.build, self.hair, self.face, cloth]
        return ", ".join(p for p in parts if p)

    @classmethod
    def load(cls, path: Path) -> "CharacterCard":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(**raw)


class PromptBuilder:
    def __init__(self, config: dict, chars_dir: Path | None = None):
        self.cfg = config
        self.style = config.get("style", {})
        self.chars: dict[str, CharacterCard] = {}
        # 角色卡目录可由 config 指定，便于一本小说一套角色
        if chars_dir is None:
            rel = config.get("paths", {}).get("chars", "assets/chars")
            chars_dir = ROOT / rel
        if chars_dir.exists():
            for f in sorted(chars_dir.glob("*.yaml")):
                card = CharacterCard.load(f)
                self.chars[card.name] = card

    # ---------- 文本片段 ----------
    def style_prefix(self) -> str:
        return " ".join(self.style.get("positive_prefix", "").split())

    def negative(self) -> str:
        base = " ".join(self.style.get("negative", "").split())
        extra = "monochrome, greyscale, sketch, rough lines, chibi, deformed hands, text artifacts"
        return f"{base}, {extra}"

    def character(self, name: str, outfit: str | None = None) -> str:
        card = self.chars.get(name)
        if not card:
            # 未登记角色：只给最保守的描述，避免模型乱编
            return f"{name}, adult, simple modern clothing"
        # 只接受角色卡里定义过的 outfit 键，其余（如 LLM 临时编的 "grey suit"）一律忽略，
        # 否则会和 default_outfit 打架，模型在两套衣服之间摇摆。
        use = outfit if outfit in card.outfits else None
        return card.describe(use)

    @staticmethod
    def clean_action(text: str | None) -> str:
        """清洗 LLM 给的动作：去人名、只取首句、限长。

        SDXL 不认识 "Chen Yu" 这种未登录 token，留着会让模型自由发挥外貌；
        长句也会挤占 CLIP 的 77 token 预算。
        """
        if not text:
            return ""
        t = re.sub(r"\b(Chen Yu|Lin Wan|ChenYu|LinWan)\b", "", text)
        # "chat" 容易让模型理解成群像对话，改成中性的消息列表
        t = re.sub(r"\bchat\b", "message list", t, flags=re.IGNORECASE)
        t = re.sub(r"\bchatting\b", "reading messages", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+", " ", t).strip(" ,.;")
        # 只保留第一句，最多 90 字符
        first = re.split(r"(?<=[.!?])\s+", t)[0]
        return first[:90].strip(" ,.;")

    def shot(self, shot_type: str) -> str:
        return SHOT_TYPES.get(shot_type, SHOT_TYPES["medium"])

    def mood(self, mood: str) -> str:
        return MOODS.get(mood, "")

    # ---------- 组装 ----------
    def build(self, shot: dict) -> tuple[str, str]:
        """shot 字段见 storyboard.py 的输出契约。

        返回 (positive, negative)。
        """
        parts: list[str] = [self.style_prefix()]
        st = shot.get("shot_type", "medium")
        parts.append(self.shot(st))
        hint = COMPOSITION_HINTS.get(st)
        if hint:
            parts.append(hint)

        # 角色：保持顺序稳定（主角在前），描述逐字复用角色卡
        for ch in shot.get("characters", []):
            name = ch.get("name", "")
            parts.append(self.character(name, ch.get("outfit")))
            if ch.get("expression"):
                parts.append(f"{ch['expression']} expression")
            if ch.get("pose"):
                parts.append(ch["pose"])

        if shot.get("action"):
            cleaned = self.clean_action(shot["action"])
            if cleaned:
                parts.append(cleaned)
        if shot.get("setting"):
            parts.append(shot["setting"])
            # 环境必须具体：防止模型画成抽象色块背景
            parts.append("detailed environment, realistic architecture, sense of place")
        if shot.get("props"):
            parts.append(", ".join(shot["props"]))
        if shot.get("mood"):
            parts.append(self.mood(shot["mood"]))
        if shot.get("lighting"):
            parts.append(shot["lighting"])
        if shot.get("extra"):
            parts.append(shot["extra"])

        # 画质尾缀
        parts.append("detailed background, sharp focus, high detail, 8k")

        positive = ", ".join(p.strip(" ,") for p in parts if p and p.strip())
        return positive, self.negative()

    def build_batch(self, shots: list[dict]) -> list[dict]:
        out = []
        for s in shots:
            pos, neg = self.build(s)
            out.append({"shot": s, "positive": pos, "negative": neg})
        return out


def main() -> None:
    ap_cfg = ROOT / "config.yaml"
    cfg = yaml.safe_load(ap_cfg.read_text(encoding="utf-8"))
    pb = PromptBuilder(cfg)
    demo = {
        "shot_type": "close",
        "characters": [{"name": "陈屿", "expression": "restrained", "pose": "standing still"}],
        "action": "holding a white envelope, looking down at it",
        "setting": "dim bedroom, bookshelf in background",
        "props": ["white envelope", "worn book"],
        "mood": "怀旧",
    }
    pos, neg = pb.build(demo)
    print("POS:", pos)
    print()
    print("NEG:", neg)
    print()
    print("已登记角色:", list(pb.chars))


if __name__ == "__main__":
    main()
