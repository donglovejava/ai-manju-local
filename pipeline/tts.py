"""台词/旁白 → wav。

后端可插拔：
  * indextts2 —— 目标方案，中文情感最好，本地跑（需先安装，见 docs/install-tts.md）
  * edge       —— 过渡方案，微软在线音色，用来先验证音画同步（需网络）
  * silent     —— 兜底，按字数生成等长静音，保证时间轴能跑通

统一约定：一个 shot 一条 wav，文件名 = shot_id.wav，时长写入 durations.json。
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHARS_PER_SEC = 4.6


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


def speech_text(shot: dict) -> str:
    """一个镜头要念的字：优先台词，其次旁白。"""
    if shot.get("dialogue"):
        return shot["dialogue"].get("text", "")
    return shot.get("narration") or ""


class SilentTTS:
    """按字数造静音，长度与真人语速一致，用于纯画面联调。"""

    def synth(self, text: str, out: Path, voice: str | None = None) -> Path:
        dots = len(re.sub(r"\s", "", text))
        dur = max(0.6, dots / CHARS_PER_SEC)
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
             "-t", f"{dur:.2f}", "-c:a", "pcm_s16le", str(out)],
            check=True, capture_output=True,
        )
        return out


class EdgeTTS:
    """过渡方案：中文神经音色，质量稳定，离线后可整体替换。"""

    VOICES = {
        "narrator": "zh-CN-YunxiNeural",   # 男声旁白
        "陈屿": "zh-CN-YunxiNeural",
        "林晚": "zh-CN-XiaoyiNeural",
        "旁白": "zh-CN-YunxiNeural",
    }

    def synth(self, text: str, out: Path, voice: str | None = None) -> Path:
        import edge_tts

        v = self.VOICES.get(voice or "narrator", "zh-CN-YunxiNeural")
        asyncio.run(edge_tts.Communicate(text, v).save(str(out)))
        return out


class IndexTTS2:
    """本地 IndexTTS-2：用参考音频做零样本音色克隆 + 情感迁移。

    依赖见 docs/install-tts.md；这里通过其官方 infer 接口调用，
    避免把 torch 依赖塞进主流水线的 Python 环境。
    """

    def __init__(self, root: Path, ref_audio: Path | None = None):
        self.root = root
        self.ref = ref_audio

    def synth(self, text: str, out: Path, voice: str | None = None) -> Path:
        ref = self.ref
        if voice and (ROOT / "assets" / "voice" / f"{voice}.wav").exists():
            ref = ROOT / "assets" / "voice" / f"{voice}.wav"
        if not ref or not ref.exists():
            raise FileNotFoundError(f"缺少参考音频: {ref}")

        cmd = [
            "python", str(self.root / "indextts" / "infer.py"),
            "--text", text,
            "--ref_audio", str(ref),
            "--output", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"IndexTTS-2 失败: {proc.stderr[-800:]}")
        return out


class ChatTTSEngine:
    """本地 ChatTTS：中文自然，纯 CPU 也能跑，音色用固定 speaker embedding。

    模型首次运行会从 HuggingFace 下载（约 1GB）；国内建议设
    HF_ENDPOINT=https://hf-mirror.com。
    """

    _chat = None  # 进程内复用，避免每句都重载模型
    _spk = None

    def __init__(self):
        import os

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        import ChatTTS

        if ChatTTSEngine._chat is None:
            chat = ChatTTS.Chat()
            chat.load(compile=False)
            ChatTTSEngine._chat = chat
            ChatTTSEngine._spk = chat.sample_random_speaker()
        self.chat = ChatTTSEngine._chat
        self.spk = ChatTTSEngine._spk

    def synth(self, text: str, out: Path, voice: str | None = None) -> Path:
        import numpy as np
        import torchaudio

        params = self.chat.InferCodeParams(
            spk_emb=self.spk,
            temperature=0.3,
            top_P=0.7,
            top_K=20,
        )
        wavs = self.chat.infer([text], params_infer_code=params, split_text=False)
        wav = np.asarray(wavs[0])
        torchaudio.save(str(out), __import__("torch").from_numpy(wav).unsqueeze(0), 24000)
        return out


class SapiTTS:
    """Windows 本地 SAPI（零依赖兜底）。

    音质一般，但完全离线、无需下载，适合验证流程。
    """

    VOICES = {
        "narrator": "Microsoft Huihui Desktop",
        "陈屿": "Microsoft Huihui Desktop",
        "林晚": "Microsoft Huihui Desktop",
        "沈迟": "Microsoft Huihui Desktop",
        "苏锦": "Microsoft Huihui Desktop",
    }

    def synth(self, text: str, out: Path, voice: str | None = None) -> Path:
        v = self.VOICES.get(voice or "narrator", "Microsoft Huihui Desktop")
        out = out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        ps = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SelectVoice('{v}'); "
            f"$s.SetOutputToWaveFile('{out}'); "
            "$s.Speak([Console]::In.ReadToEnd()); $s.Dispose()"
        )
        proc = subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command", script],
            input=text, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"SAPI TTS 失败: rc={proc.returncode} {proc.stderr[-400:]}")
        return out


class PiperTTS:
    """Piper 本地 TTS：ONNX 推理，**不需要 torch**，中文模型仅 63MB。

    依赖：
      pip install "piper-tts[zh]" unicode_rbnf
      D:/manju/TTS/piper/zh_CN-chaowen-medium.onnx (+ .json)
      D:/manju/TTS/piper/g2pW/              中文 G2P 数据（108MB 解压）
    首次运行会从 HF 拉 bert-base-chinese 分词器，之后走缓存。
    """

    _voice = None
    _voice_name: str | None = None
    DEFAULT_MODEL = "zh_CN-chaowen-medium"

    def __init__(self, root: Path | None = None, model_name: str | None = None):
        import os

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        self.root = root or Path(r"D:\manju\TTS\piper")
        self.model_path = self.root / f"{model_name or self.DEFAULT_MODEL}.onnx"

    def _ensure_voice(self):
        # 缓存要带 model_name：换音色时必须重新加载，否则还用着上一个模型
        if PiperTTS._voice is None or PiperTTS._voice_name != self.model_path.name:
            from piper import PiperVoice

            if not self.model_path.exists():
                raise FileNotFoundError(f"缺 Piper 模型: {self.model_path}")
            PiperTTS._voice = PiperVoice.load(str(self.model_path), download_dir=self.root)
            PiperTTS._voice_name = self.model_path.name
        return PiperTTS._voice

    def synth(self, text: str, out: Path, voice: str | None = None) -> Path:
        import wave

        out = out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        v = self._ensure_voice()
        # length_scale < 1 让语速略快，减少「逐字念稿」的机械感；
        # noise_scale 稍微提高一点，音色更自然（实测 0.667 偏平）。
        from piper import SynthesisConfig

        syn = SynthesisConfig(length_scale=0.92, noise_scale=0.72, noise_w_scale=0.85)
        # Piper 偶尔对某句生成 0 个音频样本，wave.close() 会抛 "# channels not specified"。
        # 这个异常发生在 with 退出时，必须捕获；重试仍失败就用静音兜底，不能让整集卡死。
        for attempt in range(2):
            try:
                with wave.open(str(out), "wb") as w:
                    v.synthesize_wav(text, w, syn_config=syn)
                if out.stat().st_size > 1024:
                    return out
                raise RuntimeError(f"输出仅 {out.stat().st_size} 字节")
            except Exception as exc:
                print(f"    Piper 合成异常（{exc}），重试 {attempt + 1}/2", flush=True)
                out.unlink(missing_ok=True)

        print(f"    Piper 持续失败，{out.name} 用静音兜底（text={text[:20]!r}）", flush=True)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(b"\x00\x00" * int(22050 * 1.5))
        return out


def make_tts(cfg: dict, force_engine: str | None = None):
    engine = force_engine or cfg["tts"]["engine"]
    if engine == "edge":
        return EdgeTTS()
    if engine == "piper":
        return PiperTTS(model_name=cfg.get("tts", {}).get("model_name"))
    if engine == "chattts":
        return ChatTTSEngine()
    if engine == "sapi":
        return SapiTTS()
    if engine == "indextts2":
        root = Path(cfg["paths"].get("indextts_root", "D:/manju/TTS/index-tts"))
        return IndexTTS2(root)
    return SilentTTS()


def run(cfg: dict, chapter: int, out_dir: Path, engine: str | None = None) -> dict:
    shots_path = Path(cfg["paths"]["work"]) / f"shots_ch{chapter:03d}.json"
    data = json.loads(shots_path.read_text(encoding="utf-8"))
    shots = data["shots"]
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = make_tts(cfg, engine)
    durations: dict[str, float] = {}
    for i, shot in enumerate(shots, 1):
        sid = shot["id"]
        wav = out_dir / f"{sid}.wav"
        text = speech_text(shot)
        if wav.exists():
            durations[sid] = probe_duration(wav)
            continue
        if not text:
            # 无字镜头：纯静音，时长用镜头设定值
            dur = float(shot.get("duration") or 3.0)
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                 "-t", f"{dur:.2f}", "-c:a", "pcm_s16le", str(wav)],
                check=True, capture_output=True,
            )
        else:
            voice = (shot.get("dialogue") or {}).get("speaker") or "narrator"
            tts.synth(text, wav, voice)
        durations[sid] = probe_duration(wav)
        print(f"  [{i}/{len(shots)}] {sid} {durations[sid]:.2f}s  {text[:24]}")

    meta = {"engine": type(tts).__name__, "durations": durations}
    (out_dir / "durations.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"配音完成 {len(durations)} 条 → {out_dir}")
    return meta


if __name__ == "__main__":
    import argparse

    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--chapter", type=int, default=1)
    ap.add_argument("--engine", default=None, help="silent|edge|indextts2")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(cfg, args.chapter, Path(cfg["paths"]["work"]) / f"ch{args.chapter:03d}" / "audio", args.engine)


def trim_silence(wav: Path, threshold_db: int = 45, tail: float = 0.12) -> float:
    """裁掉配音首尾的静音，保留句中停顿。返回裁剪后的时长。

    每个镜头一条 wav，若首尾各带 0.3-0.5s 静音，加上镜头间 padding，
    整集就会变成「一句一顿」——这是「配音渗人」的主因。
    """
    tmp = wav.with_suffix(".trim.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(wav),
         "-af", (
             "silenceremove="
             "start_periods=1:start_duration=0:start_threshold=-45dB:"
             f"stop_periods=1:stop_duration={tail}:stop_threshold=-45dB"
         ),
         str(tmp)],
        check=True, capture_output=True,
    )
    tmp.replace(wav)
    return probe_duration(wav)
