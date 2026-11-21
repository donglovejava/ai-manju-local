"""Wan 2.2 TI2V-5B 文生视频——跑通「几秒 3D」的最小验证。

依赖：
  models/unet/Wan2.2-TI2V-5B-Q4_K_M.gguf        (GGUF 量化，约 3.3GB)
  models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors (约 6.7GB)
  models/vae/Wan2.2_VAE.safetensors              (约 1.4GB)
  custom_nodes/ComfyUI-GGUF                      (UnetLoaderGGUF)

用法：
  python pipeline/wan_test.py --prompt "..." --seconds 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from comfy_client import ComfyClient

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_NEG = (
    "blurry, low quality, worst quality, jpeg artifacts, watermark, text, "
    "deformed, disfigured, bad anatomy, extra limbs, static image, slideshow"
)

# 用户要的是「漫画 3D 人物」，不是真人。三套预设分别对应三种观感。
STYLE_PRESETS: dict[str, tuple[str, str]] = {
    # 3D 国漫：像《斗罗大陆》《凡人修仙传》那种渲染
    "donghua3d": (
        "3D Chinese donghua animation, stylized CGI character, smooth rendered skin, "
        "large expressive anime eyes, semi-realistic 3D model, cel-shaded lighting, "
        "clean 3D render, high quality 3D animation",
        "photorealistic, realistic, live action, real person, photograph, film still, "
        "skin pores, wrinkles, real human face, natural lighting",
    ),
    # 三渲二：3D 模型渲染成 2D 卡通效果
    "toon3d": (
        "cel-shaded 3D anime, toon shading, 3D model rendered with flat colors and clean outlines, "
        "anime style 3D character, stylized proportions, bold rim light",
        "photorealistic, realistic, live action, real person, photograph, gradient shading, "
        "soft shadows, real human face, skin texture",
    ),
    # 皮克斯卡通
    "pixar": (
        "3D animated movie style, stylized cartoon character, soft rounded features, "
        "big expressive eyes, warm cinematic lighting, CGI render, colorful",
        "photorealistic, realistic, live action, real person, photograph, anime, 2d, flat shading, "
        "real human face, skin pores",
    ),
}


def style_prompt(style: str, subject: str) -> tuple[str, str]:
    """返回 (positive, negative)。subject 是场景描述。"""
    prefix, neg_extra = STYLE_PRESETS[style]
    return f"{prefix}, {subject}", f"{DEFAULT_NEG}, {neg_extra}"


def build_wan_workflow(
    *,
    positive: str,
    negative: str = DEFAULT_NEG,
    width: int = 704,
    height: int = 1280,
    length: int = 49,
    fps: int = 24,
    steps: int = 30,
    cfg: float = 5.0,
    seed: int = 42,
    unet: str = "Wan2.2-TI2V-5B-Q4_K_M.gguf",
    clip: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    vae: str = "Wan2.2_VAE.safetensors",
    prefix: str = "wan_test",
) -> dict:
    """Wan 2.2 5B 文生视频的 API-format workflow。

    节点编号：
      1 UnetLoaderGGUF  2 CLIPLoader  3 VAELoader
      4/5 正/负向文本   6 Wan22ImageToVideoLatent
      7 KSampler        8 VAEDecode   9 SaveWEBM
    """
    return {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": unet},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip, "type": "wan"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": positive},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": negative},
        },
        "6": {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": ["3", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
        },
        "9": {
            "class_type": "SaveWEBM",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": prefix,
                "codec": "vp9",
                "fps": float(fps),
                "crf": 28.0,
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=DEFAULT_NEG)
    ap.add_argument("--width", type=int, default=704)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="wan_test")
    args = ap.parse_args()

    # length 必须是 4n+1
    length = 4 * round((args.seconds * args.fps - 1) / 4) + 1
    client = ComfyClient(timeout=5400)  # 视频生成可能几十分钟
    if not client.is_up():
        raise SystemExit("ComfyUI 未启动")

    wf = build_wan_workflow(
        positive=args.prompt,
        negative=args.negative,
        width=args.width,
        height=args.height,
        length=length,
        fps=args.fps,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        prefix=args.prefix,
    )
    print(f"提交任务：{args.width}x{args.height}, {length} 帧 @ {args.fps}fps ≈ {length/args.fps:.1f}s, {args.steps} 步")
    t0 = time.time()
    pid = client.queue(wf)
    print(f"prompt_id = {pid}\n等待生成（8GB 显存预计 5–20 分钟）…")
    entry = client.wait(pid, poll=5.0)
    print(f"完成，耗时 {time.time()-t0:.0f}s")

    out_dir = ROOT / "work" / "wan"
    files = client.download_images(entry, out_dir, args.prefix)
    for f in files:
        print(f"输出: {f}  {f.stat().st_size/1e6:.1f}MB")


if __name__ == "__main__":
    main()
