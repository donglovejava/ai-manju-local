"""场景 → 分镜 JSON。

两条路径：
  * LLM 路径：调 OpenAI 兼容 API（默认 DeepSeek），把一段原文转成结构化镜头。
  * 规则路径：无 API key 时的降级方案——一句一镜，保证流水线始终能跑通。

契约（下游 prompt_builder / render 都依赖）：
  shot = {
    id, source_line, shot_type, characters[{name,expression,pose,outfit}],
    action, setting, props[], mood, dialogue{speaker,text} | None,
    narration, duration, transition
  }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv() -> None:
    """从项目根的 .env 读取密钥，避免把 key 写进代码或命令行。"""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

SYSTEM_PROMPT = """你是资深漫剧分镜师，把中文小说原文拆成可直接生产的镜头表。

硬性规则：
1. 台词与旁白必须逐字取自原文，不得改写、不得增删字。
2. 第一人称「我」的叙述 → narration（画外音旁白）。
3. 引号/「说：」之后的内容 → dialogue，并填 speaker（用角色名，或「旁白」）。
4. 一句原文至少 1 镜；动作密集或情绪转折处可拆 2-3 镜，但不要超过 3 镜。
5. 竖屏 9:16：以 close / medium / extreme_close 为主，wide 只用于交代环境。
6. 每镜时长 2.5-4.0 秒；台词镜按 字数/4.6 秒估算，可略长。
7. 不要凭空创造原文没有的人物、道具、地点。
8. 镜头之间默认硬切；情绪断点用 fade_black，时间跳跃用 dissolve。

【最重要的出图约束】渲染模型只懂英文。以下字段必须全部用**英文**写，不得出现中文：
  action（动作，一句话，写清人物在做什么、手在哪、视线看哪）
  setting（环境，英文短语，如 "open plan office, desk, cool fluorescent light"）
  props（英文名词数组，如 ["smartphone", "coffee cup"]）
  characters[].expression / pose / outfit（英文，如 "restrained", "sitting at desk", "grey wool coat"）
而以下字段必须保持**中文原文逐字不变**：source_line、narration、dialogue.text。

9. action 必须是**镜头能拍到的具体动作 + 具体道具**，禁止只写眼神、表情、心理活动。
   反例：eyes soft and distant / looking at the screen / recalling the past / expression falls
   正例：his fingertip presses the laptop keyboard, a document with a blinking cursor on screen
         holding a phone with both hands, thumb scrolling a message list
10. setting 必须包含**具体地点 + 至少一个可见物体 + 一种光源**。
   反例：bedroom, night
   正例：university dorm room, wooden desk with stacked books, warm desk lamp, night
11. **同一镜头只允许一个角色出现正脸。** 两人对话时：
    - 用 over_shoulder（过肩：前景一人肩膀虚化，背景一人侧脸）
    - 或拆成正反打（A 单独一镜 → B 单独一镜）
    - 不要让两个角色同框对视；characters 数组里超过一个名字时，shot_type 必须是 over_shoulder
12. **第 1 个镜头必须是钩子**：悬念、冲突、反常画面或强情绪，让人想看完。
    禁止用"这五年我做过很多事"这类平淡概括句开场。若原文开头平淡，
    把本块里最有画面冲击力的一句提到第 1 镜（旁白仍用原句，不得改写）。

只输出 JSON，不要任何解释文字。格式：
{"shots":[{"id":"001","source_line":"...","shot_type":"close|medium|wide|extreme_close|extreme_wide|over_shoulder|pov","characters":[{"name":"陈屿","expression":"calm","pose":"sitting at desk","outfit":"grey wool coat"}],"action":"holding a smartphone, looking down at the screen","setting":"open plan office, desk, cool fluorescent light","props":["smartphone"],"mood":"平静|压抑|紧张|温柔|悲伤|心动|尴尬|怀旧|夜晚|雨天|雪","dialogue":{"speaker":"林晚","text":"..."},"narration":"...","duration":3.0,"transition":"cut|fade_black|dissolve"}]}"""


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _scene_chars(sc: dict) -> int:
    if "chars" in sc:
        return sc["chars"]
    return sum(len(line) for line in sc["lines"])


def split_scene(sc: dict, max_chars: int) -> list[dict]:
    """把超长场景按行切成若干子场景。

    有些小说一章只有两三个场景，但每个场景上千字；不拆的话单次 LLM
    输出会超出 token 上限导致 JSON 截断。
    """
    out, buf, n = [], [], 0
    for line in sc["lines"]:
        L = len(line)
        if buf and n + L > max_chars:
            out.append({**sc, "lines": buf, "chars": n})
            buf, n = [], 0
        buf.append(line)
        n += L
    if buf:
        out.append({**sc, "lines": buf, "chars": n})
    return out


def chunk_scenes(chapter: dict, max_chars: int) -> list[dict]:
    """把一章的场景按字数聚合成若干块，超长场景先按行拆开。"""
    flat: list[dict] = []
    for sc in chapter["scenes"]:
        if _scene_chars(sc) > max_chars:
            flat.extend(split_scene(sc, max_chars))
        else:
            flat.append({**sc, "chars": _scene_chars(sc)})

    chunks, cur, cur_chars = [], [], 0
    for sc in flat:
        n = sc["chars"]
        if cur and cur_chars + n > max_chars:
            chunks.append({"chapter": chapter["index"], "scenes": cur})
            cur, cur_chars = [], 0
        cur.append(sc)
        cur_chars += n
    if cur:
        chunks.append({"chapter": chapter["index"], "scenes": cur})
    return chunks


def build_user_prompt(chunk: dict, char_names: list[str]) -> str:
    lines = []
    lines.append(f"【章节】第{chunk['chapter']}章")
    for sc in chunk["scenes"]:
        head = sc.get("heading") or "（无标题场景）"
        lines.append(f"\n（{head}）")
        lines.extend(sc["lines"])
    body = "\n".join(lines)
    return (
        f"出场角色（只能使用这些名字）：{', '.join(char_names)}\n\n"
        f"原文：\n{body}\n\n"
        f"请输出镜头表 JSON。"
    )


# ---------- LLM 路径 ----------
def call_llm(cfg: dict, system: str, user: str) -> dict:
    llm = cfg["llm"]
    key = os.environ.get(llm["api_key_env"], "")
    if not key:
        raise RuntimeError(f"缺少环境变量 {llm['api_key_env']}")

    payload = {
        "model": llm["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": llm.get("temperature", 0.7),
        "max_tokens": llm.get("max_tokens", 8000),
        "response_format": {"type": "json_object"},
    }
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.post(
                f"{llm['base_url'].rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return parse_json_loose(content)
        except Exception as e:  # noqa: BLE001 - 网络/解析错误统一重试
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM 调用失败: {last_err}")


def parse_json_loose(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def validate_shots(shots: list[dict], source_text: str) -> list[dict]:
    """防止模型改写台词：任何 dialogue/narration 必须能在原文里找到（去空白后包含）。"""
    flat = re.sub(r"\s", "", source_text)
    fixed = 0
    for s in shots:
        for field in ("narration",):
            t = s.get(field) or ""
            if t and re.sub(r"\s", "", t) not in flat:
                s[field] = ""
                fixed += 1
        d = s.get("dialogue") or {}
        if d.get("text") and re.sub(r"\s", "", d["text"]) not in flat:
            s["dialogue"] = None
            fixed += 1
    if fixed:
        print(f"  [校验] 剔除 {fixed} 处模型改写的文本")
    return shots


# ---------- 规则降级路径 ----------
DIALOGUE_RE = re.compile(r"^(?P<speaker>[\u4e00-\u9fa5]{1,4})?说[：:]\s*(?P<text>.+)$")

# SDXL 只认英文。规则模式必须把中文场景/道具翻成视觉短语，否则提示词等于噪声。
HEADING_MAP = {
    "通知": "open plan office, desk with laptop and smartphone, cool fluorescent light",
    "衣柜": "bedroom, open wooden wardrobe, an old coat hanging inside",
    "路上": "inside a moving taxi, city street sliding past the window, dusk",
    "饭店": "chinese restaurant interior, round tables pushed together, warm hanging lamps",
    "桌上": "round dining table, dishes, teacups, crowded restaurant",
    "有人提起": "restaurant table, people talking softly in background",
    "信封": "plain white envelope on a wooden desk, dim room",
    "我": "quiet interior, shallow depth of field",
    "照片": "old photograph held in hand, soft window light",
    "教室": "high school classroom, blackboard, rows of desks, afternoon light",
    "黑板": "chalk blackboard, dust in the air",
    "校门口": "school gate, red lanterns, winter afternoon",
    "出版社": "publishing house office, bookshelves, manuscripts on desk",
    "火车": "train compartment, window seat, landscape passing",
    "雪": "snow covered street, cold white light, breath vapor",
    "走廊": "long corridor, door at the end",
    "门": "closed door, hand on the handle",
}

PROP_MAP = {
    "手机": "smartphone",
    "信封": "white envelope",
    "信": "folded letter",
    "书": "worn book",
    "书架": "bookshelf",
    "镜子": "mirror",
    "窗": "window",
    "雨": "rain on glass",
    "雪": "falling snow",
    "黑板": "blackboard",
    "保温杯": "thermos",
    "相机": "vintage camera",
    "照片": "old photograph",
    "毛衣": "knitted sweater",
    "外套": "wool coat",
    "车": "car",
    "桌子": "table",
    "面": "bowl of noodles",
}

ACTION_MAP = {
    "亮了一下": "phone screen lighting up",
    "划开屏幕": "swiping phone screen",
    "看了一眼": "glancing briefly",
    "关了屏幕": "turning off the screen",
    "点开": "tapping the screen",
    "扣在桌上": "placing phone face down",
    "穿上": "putting on a coat",
    "扣上扣子": "buttoning up",
    "闻了一下": "smelling the fabric",
    "推门": "pushing the door open",
    "坐下": "sitting down",
    "拆": "opening an envelope",
    "折": "folding a piece of paper",
    "放回": "putting it back",
}


def pick_visual(heading: str, line: str) -> tuple[str, str | None, list[str]]:
    """从场景标题和原文里抽可视元素 → (setting, action, props)。"""
    setting = HEADING_MAP.get(heading, "modern chinese interior, natural light")
    action = next((v for k, v in ACTION_MAP.items() if k in line), None)
    props = [v for k, v in PROP_MAP.items() if k in line][:3]
    # 标题里带道具时也补上
    for k, v in PROP_MAP.items():
        if k in heading and v not in props:
            props.append(v)
    return setting, action, props


def rule_based_shots(scenes: list[dict], char_names: list[str]) -> list[dict]:
    shots: list[dict] = []
    idx = 0
    for sc in scenes:
        heading = sc.get("heading") or ""
        for line in sc["lines"]:
            idx += 1
            line = line.strip()
            if not line:
                continue
            n = len(line)
            m = DIALOGUE_RE.match(line)
            dialogue, narration = None, None
            if m:
                speaker = m.group("speaker") or "旁白"
                dialogue = {"speaker": speaker, "text": m.group("text")}
            else:
                narration = line

            setting, action, props = pick_visual(heading, line)

            # 镜头类型按句子长度粗分：极短句=特写细节，长句=中景
            if n <= 6:
                shot_type = "extreme_close"
            elif n <= 18:
                shot_type = "close"
            elif n <= 34:
                shot_type = "medium"
            else:
                shot_type = "wide"

            dur = max(2.5, min(4.0, (n / 4.6) + 0.8))
            shots.append(
                {
                    "id": f"{idx:03d}",
                    "source_line": line,
                    "shot_type": shot_type,
                    "characters": [{"name": char_names[0], "expression": "restrained", "pose": None, "outfit": None}],
                    "action": action,
                    "setting": setting,
                    "props": props,
                    "mood": "平静",
                    "dialogue": dialogue,
                    "narration": narration,
                    "duration": round(dur, 2),
                    "transition": "cut",
                }
            )
    return shots


# ---------- 主流程 ----------
def run(cfg: dict, chapter_index: int, use_llm: bool, limit_scenes: int | None) -> dict:
    load_dotenv()
    scenes_path = Path(cfg["paths"]["work"]) / "scenes.json"
    data = json.loads(scenes_path.read_text(encoding="utf-8"))
    chapter = next(c for c in data["chapters"] if c["index"] == chapter_index)
    if limit_scenes:
        chapter = {**chapter, "scenes": chapter["scenes"][:limit_scenes]}

    char_names = cfg.get("characters") or ["陈屿", "林晚"]
    all_shots: list[dict] = []

    if use_llm:
        chunks = chunk_scenes(chapter, cfg["llm"]["chunk_chars"])
        print(f"第{chapter_index}章：{len(chapter['scenes'])} 场景 → {len(chunks)} 块")
        for i, ch in enumerate(chunks, 1):
            user = build_user_prompt(ch, char_names)
            print(f"  块 {i}/{len(chunks)}（{sum(s['chars'] for s in ch['scenes'])} 字）…")
            try:
                result = call_llm(cfg, SYSTEM_PROMPT, user)
                shots = result.get("shots", [])
                src = "\n".join("\n".join(s["lines"]) for s in ch["scenes"])
                shots = validate_shots(shots, src)
                all_shots.extend(shots)
            except Exception as e:  # noqa: BLE001
                print(f"  LLM 失败，回退规则分镜: {e}")
                all_shots.extend(rule_based_shots(ch["scenes"], char_names))
    else:
        all_shots = rule_based_shots(chapter["scenes"], char_names)

    # 重新编号，保证全章连续
    for i, s in enumerate(all_shots, 1):
        s["id"] = f"{i:03d}"

    src_chars = sum(s["chars"] for s in chapter["scenes"])
    total_dur = sum(float(s.get("duration") or 3.0) for s in all_shots)
    out = {
        "chapter": chapter_index,
        "title": chapter["title"],
        "pov": chapter["pov"],
        "source_chars": src_chars,
        "shots": all_shots,
        "stats": {
            "shots": len(all_shots),
            "duration_sec": round(total_dur, 1),
            "avg_shot_sec": round(total_dur / max(1, len(all_shots)), 2),
        },
    }
    out_path = Path(cfg["paths"]["work"]) / f"shots_ch{chapter_index:03d}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n镜头数 {len(all_shots)}，总时长 {total_dur/60:.1f} 分钟，已写 {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--chapter", type=int, default=1)
    ap.add_argument("--llm", action="store_true", help="使用 LLM 分镜；不给则用规则降级")
    ap.add_argument("--limit-scenes", type=int, default=0, help="只处理前 N 个场景（调试用）")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    run(cfg, args.chapter, args.llm, args.limit_scenes or None)


if __name__ == "__main__":
    main()
