"""端到端编排：shots.json → 图 → 音 → 镜头 → 成片。

用法：
  python pipeline/run_episode.py --chapter 1 --max-shots 20 --tts silent
  python pipeline/run_episode.py --chapter 1 --max-shots 20 --tts edge --skip-images
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from comfy_client import ComfyClient, build_sdxl_workflow
from prompt_builder import PromptBuilder
from render import (
    Clip,
    build_ass,
    burn_subtitles,
    concat_and_mix,
    make_ambience,
    pick_ambience,
    probe_duration,
    render_shot,
    run_ffmpeg,
)
from tts import make_tts, speech_text, trim_silence

ROOT = Path(__file__).resolve().parent.parent

# 角色名 → ComfyUI/input 下的参考图（IP-Adapter 锁脸用）
REF_MAP = {
    "陈屿": "ref_chenyu.png",
    "林晚": "ref_linwan.png",
}

# 角色 → 专属 LoRA。一个镜头里出现多个角色时，取第一个（通常是主体/前景）。
LORA_MAP = {
    "沈迟": "shenchi_v3.safetensors",
    "苏锦": "sujin_v2.safetensors",
}


def pick_lora(shot: dict, default: str | None) -> str | None:
    """按镜头里的第一个已登记角色选 LoRA；未登记则回退到命令行传入的默认值。"""
    for ch in shot.get("characters", []):
        name = ch.get("name")
        if name in LORA_MAP:
            return LORA_MAP[name]
    return default


def load_shots(cfg: dict, chapter: int, max_shots: int, shots_file: str | None = None) -> dict:
    p = Path(shots_file) if shots_file else Path(cfg["paths"]["work"]) / f"shots_ch{chapter:03d}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    if max_shots:
        data["shots"] = data["shots"][:max_shots]
    return data


def pick_ref(shot: dict) -> str | None:
    """按出场角色选 IP-Adapter 参考图（文件名相对 ComfyUI/input）。"""
    for ch in shot.get("characters", []):
        ref = REF_MAP.get(ch.get("name"))
        if ref:
            return ref
    return None


def stable_seed(shot_id: str, salt: int = 0) -> int:
    """由镜头号派生固定 seed：同一镜重跑结果可复现，坏图才好针对性重抽。"""
    h = hashlib.md5(f"{shot_id}:{salt}".encode()).hexdigest()
    return int(h[:8], 16) % (2**32)


def shot_lora_strength(img_cfg: dict, shot: dict, override: float | None = None) -> float:
    """按景别给 LoRA 强度：近景压低防幼态，全景拉高锁形象。

    实测 0.6 在近景能避免幼态，但全身/中景锁不住脸，所以按 shot_type 分档。
    """
    if override is not None:
        return override
    table = img_cfg.get("lora_strength_by_shot") or {}
    return float(table.get(shot.get("shot_type", "medium"), img_cfg.get("lora_strength_default", 0.7)))


def gen_images(
    cfg: dict,
    shots: list[dict],
    img_dir: Path,
    ckpt: str,
    lora: str | None,
    use_ref: bool = False,
    ref_weight: float = 0.8,
    lora_strength: float | None = None,
) -> list[Path]:
    client = ComfyClient()
    if not client.is_up():
        raise RuntimeError("ComfyUI 未启动：先运行 D:/manju/ComfyUI/run_nvidia_gpu.bat")
    pb = PromptBuilder(cfg)
    img_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    img_cfg = cfg["image"]
    for i, shot in enumerate(shots, 1):
        dst = img_dir / f"{shot['id']}.png"
        if dst.exists():
            out.append(dst)
            continue
        pos, neg = pb.build(shot)
        ref = pick_ref(shot) if use_ref else None
        lora_name = pick_lora(shot, lora)
        strength = shot_lora_strength(img_cfg, shot, lora_strength)
        (img_dir / f"{shot['id']}.prompt.txt").write_text(
            f"LORA: {lora_name} @ {strength}\nREF: {ref}\n{pos}\n\nNEG: {neg}", encoding="utf-8"
        )
        wf = build_sdxl_workflow(
            ckpt_name=ckpt,
            positive=pos,
            negative=neg,
            width=img_cfg["base_width"],
            height=img_cfg["base_height"],
            steps=img_cfg["steps"],
            cfg=img_cfg["cfg"],
            sampler=img_cfg["sampler"],
            scheduler=img_cfg["scheduler"],
            clip_skip=img_cfg.get("clip_skip", 2),
            lora_name=lora_name,
            lora_strength=strength,
            ipadapter_ref=ref,
            ipadapter_weight=ref_weight,
            seed=stable_seed(shot["id"]),
        )
        print(f"  [{i}/{len(shots)}] 出图 {shot['id']} {shot.get('shot_type')} lora={lora_name}@{strength:.2f} …", end="", flush=True)
        files = client.generate(wf, img_dir, f"tmp_{shot['id']}")
        if not files:
            raise RuntimeError(f"镜头 {shot['id']} 没有返回图片")
        shutil.move(str(files[0]), dst)
        for extra in files[1:]:
            extra.unlink(missing_ok=True)
        print(" ok")
        out.append(dst)
    return out


def gen_audio(cfg: dict, shots: list[dict], audio_dir: Path, engine: str | None) -> dict[str, float]:
    audio_dir.mkdir(parents=True, exist_ok=True)
    tts = make_tts(cfg, engine)
    durations: dict[str, float] = {}
    for i, shot in enumerate(shots, 1):
        sid = shot["id"]
        wav = audio_dir / f"{sid}.wav"
        text = speech_text(shot)
        # 0 字节/损坏的 wav 必须先删掉重生成：它会被当成"已存在"跳过合成，
        # 然后 trim_silence 里 ffmpeg 处理它直接崩（0xC0000409，实测 EP019 的 029.wav）。
        if wav.exists() and wav.stat().st_size < 1024:
            wav.unlink(missing_ok=True)
        if not wav.exists():
            if text:
                voice = (shot.get("dialogue") or {}).get("speaker") or "narrator"
                tts.synth(text, wav, voice)
            else:
                run_ffmpeg([
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", f"{float(shot.get('duration') or 3.0):.2f}",
                    "-c:a", "pcm_s16le", str(wav),
                ])
        # 裁掉首尾静音，避免镜头间出现长死寂（幂等，已裁过再裁不会更短）
        if text:
            durations[sid] = trim_silence(wav)
        else:
            durations[sid] = probe_duration(wav)
        print(f"  [{i}/{len(shots)}] 配音 {sid} {durations[sid]:.2f}s")
    return durations


def mux(video: Path, audio: Path, out: Path, duration: float) -> Path:
    run_ffmpeg([
        "-i", str(video), "-i", str(audio),
        # 统一响度到 -14 LUFS（平台标准），真峰值压到 -1.5 dBTP，重采样 44.1k 补足单薄感
        "-filter_complex",
        "[1:a]apad,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=44100[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", f"{duration:.3f}", str(out),
    ])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--chapter", type=int, default=1)
    ap.add_argument("--max-shots", type=int, default=20)
    ap.add_argument("--shots-file", default=None, help="直接指定分镜 JSON（批量模式）")
    ap.add_argument("--episode", type=int, default=None, help="集号，用于输出命名与标题卡")
    ap.add_argument("--tts", default=None, help="silent|edge|indextts2")
    ap.add_argument("--ckpt", default="Illustrious-XL-v2.0.safetensors")
    ap.add_argument("--lora", default=None)
    ap.add_argument(
        "--lora-strength", type=float, default=None,
        help="强制统一 LoRA 强度；不填则按 config.image.lora_strength_by_shot 按景别取值",
    )
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument(
        "--use-ref", action="store_true",
        help="启用 IP-Adapter 锁脸（实验性：当前底模+PLUS 组合会性别反转并过曝，默认关闭）",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ep_tag = f"ep{args.episode:03d}" if args.episode else f"ch{args.chapter:03d}"
    work = Path(cfg["paths"]["work"]) / ep_tag
    img_dir, audio_dir, clip_dir = work / "images", work / "audio", work / "clips"
    for d in (img_dir, audio_dir, clip_dir):
        d.mkdir(parents=True, exist_ok=True)

    data = load_shots(cfg, args.chapter, args.max_shots, args.shots_file)
    shots = data["shots"]
    print(f"第{args.chapter}章 {data['title']}  镜头 {len(shots)} 个")

    if not args.skip_images:
        gen_images(
            cfg, shots, img_dir, args.ckpt, args.lora,
            use_ref=args.use_ref, lora_strength=args.lora_strength,
        )

    if args.skip_audio:
        durations = {s["id"]: float(s.get("duration") or 3.0) for s in shots}
    else:
        durations = gen_audio(cfg, shots, audio_dir, args.tts)

    if args.skip_render:
        return

    # 每镜时长 = max(设定时长, 配音时长 + 0.35s 呼吸)
    final_durs: list[float] = []
    clips: list[Clip] = []
    from render import pick_move

    for i, shot in enumerate(shots):
        sid = shot["id"]
        audio_dur = durations.get(sid, 0.0)
        if audio_dur > 0:
            # 以配音为准：说完即切。原先是 max(LLM 时长, 音频+0.35)，
            # LLM 常给 6 秒而配音只有 2-4 秒，中间留下大段死寂（实测 56 处 >0.6s 静音）。
            # padding 从 0.4 降到 0.15：首尾静音已被 trim_silence 裁掉，无需再留大空隙。
            dur = max(1.8, audio_dur + 0.15)
        else:
            dur = float(shot.get("duration") or 3.0)
        dur = min(dur, cfg["video"]["shot_max_sec"] + 2.0)
        final_durs.append(dur)
        img = img_dir / f"{sid}.png"
        if not img.exists():
            raise FileNotFoundError(f"缺图: {img}")
        silent = clip_dir / f"{sid}_v.mp4"
        render_shot(
            img, dur, pick_move(shot, i), cfg["video"]["fps"], silent,
            transition=shot.get("transition", "cut"),
        )
        wav = audio_dir / f"{sid}.wav"
        clip = clip_dir / f"{sid}.mp4"
        if wav.exists():
            mux(silent, wav, clip, dur)
        else:
            clip = silent
        clips.append(Clip(shot_id=sid, video=clip, duration=dur, audio=wav if wav.exists() else None))
        print(f"  [{i+1}/{len(shots)}] 渲染 {sid} {dur:.2f}s")

    rough = work / f"{ep_tag}_rough.mp4"
    # 环境音：按整集场景推断类型，铺一条底噪
    total_dur = sum(final_durs)
    kind = pick_ambience(shots)
    ambience = work / f"ambience_{ep_tag}.wav"
    make_ambience(kind, total_dur + 1.0, ambience)
    print(f"  环境音: {kind}  {total_dur:.1f}s")

    # BGM：可选，放 assets/bgm/default.mp3 就会自动混入
    bgm = Path(cfg["paths"]["assets"]) / "bgm" / "default.mp3"
    if not bgm.exists():
        bgm = None

    concat_and_mix(clips, rough, cfg["video"]["fps"], bgm=bgm, ambience=ambience)

    ass = work / f"{ep_tag}.ass"
    build_ass(shots, final_durs, ass)

    final = Path(cfg["paths"]["output"]) / f"{ep_tag}.mp4"
    final.parent.mkdir(parents=True, exist_ok=True)

    # 先把字幕烧到正片，再拼 2 秒标题卡到片头
    body = work / f"{ep_tag}_body.mp4"
    burn_subtitles(rough, ass, body)

    from render import make_title_card

    card = work / "title_card.mp4"
    make_title_card(args.episode or args.chapter, data.get("title") or "", card, fps=cfg["video"]["fps"])
    # 不能 -c copy，也不能用 demuxer concat：标题卡是 30fps、正片是 29.97fps，
    # timebase 不一致会让容器 duration 翻倍（实测 330s vs 实际 169s）。
    # 用 concat demuxer 而不是 filter_complex：filter_complex 在 Windows 上偶发死循环
    # （实测 EP017 烧掉 610 秒 CPU 仍只输出 48 字节空文件），concat demuxer 秒级完成。
    # 列表里必须是 Windows 绝对路径（C:/...），Git Bash 的 /c/... 格式 ffmpeg 不认。
    concat_txt = work / f"{ep_tag}_concat.txt"
    concat_txt.write_text(
        "".join(f"file '{str(p.resolve()).replace(chr(92), '/')}'\n" for p in (card, body)),
        encoding="utf-8",
    )
    run_ffmpeg([
        "-f", "concat", "-safe", "0", "-i", str(concat_txt),
        # 两段（标题卡 + 正片）编码参数完全一致，直接 copy 流最稳。
        # 重新编码在 Windows 上偶发卡死并留下 48 字节空文件（实测 EP021）。
        "-c", "copy",
        str(final),
    ])
    concat_txt.unlink(missing_ok=True)

    print(f"\n成片: {final}  时长 {probe_duration(final):.1f}s")


if __name__ == "__main__":
    main()
