"""小说 → 章节 → 场景切分。

本项目的原文结构（作者固定写法）：
    ## 第1章 章节标题
    （场景名）
    正文一行一句……
    （下一个场景名）
    ……

第 4 章起出现 【她】 视角标记（女性视角章节），既有章标题前的
`## 第N章 【她】xxx`，也有正文中单独成行的 `【她】`。

输出 scenes.json，供 storyboard.py 消费。
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

CHAPTER_RE = re.compile(r"^##\s+(第\d+章)\s*(.*)$")
# 场景标记：整行只有 （xxx） 或 【xxx】
SCENE_RE = re.compile(r"^(?:（(?P<paren>[^）]{1,20})）|【(?P<brack>[^】]{1,20})】)$")


@dataclass
class Scene:
    scene_id: str
    heading: str
    kind: str  # "scene" 普通场景 / "shot" 分镜式短标记 / "pov" 视角标记
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def chars(self) -> int:
        return len(re.sub(r"\s", "", self.text))


@dataclass
class Chapter:
    index: int
    title: str
    raw_title: str
    pov: str | None = None
    scenes: list[Scene] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(s.chars for s in self.scenes)

    @property
    def shot_estimate(self) -> int:
        """估算镜头数：一句一镜，极小句子合并，极长句拆分。"""
        total = 0.0
        for s in self.scenes:
            for line in s.lines:
                n = len(re.sub(r"\s", "", line))
                if n == 0:
                    continue
                if n <= 8:      # 「去。」这类极短句：与相邻合并
                    total += 0.6
                elif n <= 40:   # 常规短句：一镜
                    total += 1.0
                else:           # 长句：按 40 字拆
                    total += max(1.4, n / 40)
        return int(round(total))


def parse_chapter_title(raw: str) -> tuple[str | None, str]:
    """从 '【她】我第一次见他' 里拆出视角。"""
    m = re.match(r"^【(?P<pov>[^】]+)】\s*(?P<title>.*)$", raw.strip())
    if m:
        return m.group("pov"), m.group("title").strip()
    return None, raw.strip()


def classify_heading(text: str, is_pov: bool) -> str:
    if is_pov:
        return "pov"
    # 长度 <= 4 且出现频次高的短标记（如「我」「信封」）视为分镜式节奏标记
    if len(text) <= 4:
        return "shot"
    return "scene"


def split(novel_path: Path) -> dict:
    text = novel_path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()

    chapters: list[Chapter] = []
    cur_ch: Chapter | None = None
    cur_scene: Scene | None = None

    def flush_scene():
        nonlocal cur_scene
        if cur_ch is not None and cur_scene is not None:
            # 去掉纯空白场景
            if cur_scene.lines:
                cur_ch.scenes.append(cur_scene)
        cur_scene = None

    for raw in lines:
        line = raw.rstrip()
        m = CHAPTER_RE.match(line)
        if m:
            flush_scene()
            idx = int(re.search(r"\d+", m.group(1)).group())
            pov, title = parse_chapter_title(m.group(2))
            cur_ch = Chapter(index=idx, title=title, raw_title=m.group(2).strip(), pov=pov)
            chapters.append(cur_ch)
            continue
        if cur_ch is None:
            continue

        sm = SCENE_RE.match(line.strip())
        if sm:
            flush_scene()
            if sm.group("paren"):
                name, is_pov = sm.group("paren"), False
            else:
                name, is_pov = sm.group("brack"), True
            sid = f"c{cur_ch.index:03d}_s{len(cur_ch.scenes) + 1:02d}"
            cur_scene = Scene(scene_id=sid, heading=name, kind=classify_heading(name, is_pov))
            continue

        if not line.strip():
            continue
        if cur_scene is None:
            # 章首没有场景标记时的兜底
            sid = f"c{cur_ch.index:03d}_s{len(cur_ch.scenes) + 1:02d}"
            cur_scene = Scene(scene_id=sid, heading="", kind="scene")
        cur_scene.lines.append(line.strip())

    flush_scene()

    payload = {
        "book": {
            "path": str(novel_path),
            "chapters": len(chapters),
            "chars": sum(c.chars for c in chapters),
            "shots_estimate": sum(c.shot_estimate for c in chapters),
        },
        "chapters": [
            {
                "index": c.index,
                "title": c.title,
                "pov": c.pov,
                "chars": c.chars,
                "shot_estimate": c.shot_estimate,
                "scenes": [asdict(s) | {"chars": s.chars} for s in c.scenes],
            }
            for c in chapters
        ],
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=None, help="默认写到 config.paths.work/scenes.json")
    ap.add_argument("--chapter", type=int, default=0, help="只导出某一章（0=全部）")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    novel_path = Path(cfg["novel"]["path"])
    data = split(novel_path)

    if args.chapter:
        data["chapters"] = [c for c in data["chapters"] if c["index"] == args.chapter]

    out = Path(args.out) if args.out else Path(cfg["paths"]["work"]) / "scenes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"章节数: {data['book']['chapters']}  总字数: {data['book']['chars']}")
    print(f"估算总镜头: {data['book']['shots_estimate']}  (按 3 秒/镜 ≈ {data['book']['shots_estimate'] * 3 / 60:.0f} 分钟)")
    for c in data["chapters"][:5]:
        print(f"  第{c['index']}章 {c['title']}  pov={c['pov']}  {c['chars']}字  场景{len(c['scenes'])}  镜头≈{c['shot_estimate']}")
    print(f"已写出: {out}")


if __name__ == "__main__":
    main()
