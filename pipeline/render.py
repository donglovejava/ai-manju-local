"""镜头合成：静图 + 运镜 + 字幕 + 音频 → mp4。

FFmpeg 负责全部像素工作。这里只有三件事：
  1. 把 SDXL 竖图 cover 裁切成 2160×3840，再用 zoompan 做缓动运镜
  2. 生成 ASS 字幕（竖屏安全区、描边）
  3. 拼接镜头并混音
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W, H = 1080, 1920
UP_W, UP_H = 2160, 3840  # 2x 工作分辨率，避免 zoompan 抖动


@dataclass
class Clip:
    shot_id: str
    video: Path
    duration: float
    audio: Path | None = None


def run_ffmpeg(args: list[str], *, quiet: bool = True, timeout: int = 900) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error" if quiet else "info", *args]
    # 必须带超时：ffmpeg 在 Windows 上偶发卡死（实测 EP017 卡在最后一步 filter concat），
    # 没有 timeout 时 subprocess.run 会永远等，只能靠外层看门狗 kill。
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg 超时（{timeout}s）: {' '.join(cmd[:12])} …")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败:\n{' '.join(cmd[:12])} ...\n{proc.stderr[-1500:]}")


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def zoompan_filter(move: str, frames: int, fps: int) -> str:
    """运镜表达式。on = 输出帧序号，iw/ih = zoompan 输入尺寸。"""
    n = max(frames, 1)
    # 位移留 10% 余量（zoom 固定 1.12），避免边缘露黑
    if move == "zoom_in":
        z = f"min(1+0.12*on/{n},1.12)"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif move == "zoom_out":
        z = f"max(1.12-0.12*on/{n},1.0)"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif move == "pan_left":
        z = "1.12"
        x = f"(iw-iw/zoom)*(1-on/{n})"
        y = "ih/2-(ih/zoom/2)"
    elif move == "pan_right":
        z = "1.12"
        x = f"(iw-iw/zoom)*(on/{n})"
        y = "ih/2-(ih/zoom/2)"
    elif move == "tilt_up":
        z = "1.12"
        x = "iw/2-(iw/zoom/2)"
        y = f"(ih-ih/zoom)*(1-on/{n})"
    elif move == "tilt_down":
        z = "1.12"
        x = "iw/2-(iw/zoom/2)"
        y = f"(ih-ih/zoom)*(on/{n})"
    else:  # static
        z = "1.04"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"

    return (
        f"scale={UP_W}:{UP_H}:force_original_aspect_ratio=increase,"
        f"crop={UP_W}:{UP_H},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={W}x{H}:fps={fps},"
        f"format=yuv420p"
    )


def pick_move(shot: dict, index: int) -> str:
    """按景别与情绪选运镜，避免全片一个方向。"""
    st = shot.get("shot_type", "medium")
    mood = shot.get("mood", "平静")
    if st in ("extreme_close", "close"):
        return "zoom_in" if index % 2 == 0 else "zoom_out"
    if st in ("wide", "extreme_wide"):
        return "pan_left" if index % 2 == 0 else "pan_right"
    if mood in ("紧张", "心动"):
        return "zoom_in"
    return ["pan_left", "pan_right", "tilt_up", "tilt_down"][index % 4]


def render_shot(
    image: Path,
    duration: float,
    move: str,
    fps: int,
    out: Path,
    transition: str = "cut",
) -> Path:
    frames = max(1, round(duration * fps))
    vf = zoompan_filter(move, frames, fps)
    # 转场：单镜头靠淡入淡出模拟黑场；相邻两个 fade_black 之间自然形成黑场呼吸
    if transition == "fade_black":
        vf += f",fade=t=in:st=0:d=0.25,fade=t=out:st={max(0.0, duration - 0.25):.2f}:d=0.25"
    elif transition == "dissolve":
        vf += ",fade=t=in:st=0:d=0.18"
    run_ffmpeg([
        "-loop", "1", "-i", str(image),
        "-vf", vf,
        "-t", f"{duration:.3f}",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        str(out),
    ], timeout=90)  # 单片段正常 2-5 秒；90 秒还没完成就是 ffmpeg 卡死，快速失败让上层重试
    return out


# ---------- 字幕 ----------
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Narration,Microsoft YaHei,44,&H00F2F2F2,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,1,0,1,2,1,2,80,80,160,1
Style: Dialogue,Microsoft YaHei,48,&H00FFFFFF,&H000000FF,&H00101010,&H80000000,-1,0,0,0,100,100,1,0,1,2,1,2,80,80,160,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(shots: list[dict], durations: list[float], out: Path) -> Path:
    lines = [ASS_HEADER.format(w=W, h=H)]
    t = 0.0
    for shot, dur in zip(shots, durations):
        text = ""
        style = "Narration"
        if shot.get("dialogue"):
            text = shot["dialogue"].get("text", "")
            style = "Dialogue"
        elif shot.get("narration"):
            text = shot["narration"]
            style = "Narration"
        if text:
            # ASS 转义：反斜杠、花括号、换行
            text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            lines.append(
                f"Dialogue: 0,{_ass_time(t)},{_ass_time(t + dur)},{style},,0,0,0,,{text}"
            )
        t += dur
    out.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return out


# ---------- 拼接 ----------
# 环境音：用 ffmpeg 噪声源合成，完全本地、无版权问题。
# 音量故意压得很低（0.12-0.35），只做「底噪」不做主角。
AMBIENCE = {
    "room": "anoisesrc=color=brown:amplitude=0.12:seed=11,lowpass=f=1500,volume=0.18",
    "rain": "anoisesrc=color=pink:amplitude=0.35:seed=22,highpass=f=400,lowpass=f=7000,volume=0.32",
    "wind": "anoisesrc=color=brown:amplitude=0.30:seed=33,lowpass=f=700,volume=0.28",
    "snow": "anoisesrc=color=pink:amplitude=0.22:seed=44,lowpass=f=3000,volume=0.22",
    "street": "anoisesrc=color=brown:amplitude=0.25:seed=55,bandpass=f=500:w=300,volume=0.24",
}


def make_ambience(kind: str, duration: float, out: Path) -> Path:
    """生成一条全长的环境音轨。kind 见 AMBIENCE。"""
    af = AMBIENCE.get(kind, AMBIENCE["room"])
    run_ffmpeg([
        "-f", "lavfi", "-i", af,
        "-t", f"{duration:.2f}",
        "-c:a", "pcm_s16le", str(out),
    ])
    return out


def pick_ambience(shots: list[dict]) -> str:
    """从整集的 setting 推断环境音类型，取出现最多的场景。"""
    import re

    score = {k: 0 for k in AMBIENCE}
    for s in shots:
        text = f"{s.get('setting') or ''} {s.get('mood') or ''}".lower()
        if re.search(r"rain|雨", text):
            score["rain"] += 2
        if re.search(r"snow|雪|winter|冬", text):
            score["snow"] += 2
        if re.search(r"street|road|outdoor|街|路|广场", text):
            score["street"] += 1
        if re.search(r"wind|风", text):
            score["wind"] += 1
        score["room"] += 1  # 室内兜底，每镜都加分
    return max(score, key=score.get)


def concat_and_mix(
    clips: list[Clip],
    out: Path,
    fps: int,
    bgm: Path | None = None,
    ambience: Path | None = None,
) -> Path:
    """按顺序拼接视频轨，可叠加 BGM 与环境音；台词音轨已在单镜头阶段烧入。"""
    list_file = out.with_suffix(".concat.txt")
    # 必须用绝对路径：ffmpeg 的 concat 列表是相对「列表文件所在目录」解析的
    list_file.write_text(
        "\n".join(f"file '{c.video.resolve().as_posix()}'" for c in clips), encoding="utf-8"
    )
    if (bgm and bgm.exists()) or (ambience and ambience.exists()):
        inputs = ["-f", "concat", "-safe", "0", "-i", str(list_file)]
        chains = []
        idx = 1
        labels = ["[0:a]"]
        if ambience and ambience.exists():
            inputs += ["-stream_loop", "-1", "-i", str(ambience)]
            chains.append(f"[{idx}:a]volume=1.0[amb]")
            labels.append("[amb]")
            idx += 1
        if bgm and bgm.exists():
            inputs += ["-stream_loop", "-1", "-i", str(bgm)]
            chains.append(f"[{idx}:a]volume=0.18[bg]")
            labels.append("[bg]")
            idx += 1
        mixed = "".join(labels)
        # normalize=0 关键：amix 默认按输入数衰减（2 路各 -6dB），会把台词压到 -21 LUFS
        chains.append(
            f"{mixed}amix=inputs={len(labels)}:duration=first:dropout_transition=0:normalize=0[a]"
        )
        run_ffmpeg([
            *inputs,
            "-filter_complex", ";".join(chains),
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            str(out),
        ])
    else:
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out)])
    list_file.unlink(missing_ok=True)
    return out


def burn_subtitles(video: Path, ass: Path, out: Path) -> Path:
    # Windows 路径在滤镜里要转义盘符冒号和反斜杠
    ass_arg = str(ass).replace("\\", "/").replace(":", "\\:")
    run_ffmpeg([
        "-i", str(video),
        "-vf", f"ass='{ass_arg}'",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy",
        str(out),
    ])
    return out


if __name__ == "__main__":
    import sys

    # 自检：用 ffmpeg 造一张测试图，跑一遍运镜
    tmp = ROOT / "work" / "_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    img = tmp / "test.png"
    if not img.exists():
        run_ffmpeg(["-f", "lavfi", "-i", f"testsrc2=s={UP_W}x{UP_H}", "-frames:v", "1", str(img)])
    for mv in ("zoom_in", "pan_left", "tilt_up"):
        out = tmp / f"{mv}.mp4"
        render_shot(img, 2.5, mv, 30, out)
        print(f"{mv}: {out} {probe_duration(out):.2f}s")
    print("self-test ok")


def make_title_card(
    episode: int,
    title: str,
    out: Path,
    duration: float = 2.0,
    fps: int = 30,
) -> Path:
    """生成 2 秒黑底标题卡：第N集 / 章节名。"""
    font = r"C\:/Windows/Fonts/msyh.ttc"
    # 中文字符在 drawtext 里需要转义
    ep_text = f"第 {episode} 集"
    safe_title = title.replace("'", r"\'").replace(":", r"\:")
    vf = (
        f"drawtext=fontfile='{font}':text='{ep_text}':fontcolor=white:fontsize=72:"
        f"x=(w-text_w)/2:y=h/2-120,"
        f"drawtext=fontfile='{font}':text='{safe_title}':fontcolor=0xB0B0B0:fontsize=44:"
        f"x=(w-text_w)/2:y=h/2+40"
    )
    # 带一条静音音轨，方便和正片直接 concat
    run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s=1080x1920:d={duration}:r={fps}",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(out),
    ])
    return out
