"""双角色 LoRA 测试：单人稳定性 + 双人同框。

用法：
    python pipeline/dual_lora_test.py            # 单人 4 张 + 双人 4 张
    python pipeline/dual_lora_test.py --only dual
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from comfy_client import ComfyClient
from prompt_builder import PromptBuilder

ROOT = Path(__file__).resolve().parent.parent

SINGLE = [
    ("sujin_closeup", "sujin, {sujin}, close-up of face and shoulders, looking at viewer, calm expression"),
    ("sujin_desk", "sujin, {sujin}, sitting at a wooden desk, reading a book, warm desk lamp"),
    ("sujin_window", "sujin, {sujin}, standing by a large window, looking outside, soft daylight"),
    ("sujin_street", "sujin, {sujin}, standing on a city street at night, holding a bag, three quarter view"),
]

DUAL = [
    ("dual_classroom", "shenchi, {shenchi}, sujin, {sujin}, standing face to face in a high school classroom, afternoon light"),
    ("dual_corridor", "shenchi, {shenchi}, sujin, {sujin}, walking side by side in a school corridor, both in school uniforms"),
    ("dual_desk", "shenchi, {shenchi}, sujin, {sujin}, sitting at the same desk, one reading, one writing"),
    ("dual_snow", "shenchi, {shenchi}, sujin, {sujin}, standing in falling snow at night, facing each other"),
]


def build_dual_lora_workflow(
    *, ckpt: str, positive: str, negative: str, width: int, height: int,
    steps: int, cfg: float, sampler: str, scheduler: str, seed: int,
    clip_skip: int, lora1: str, lora2: str, strength: float,
) -> dict:
    """两个 LoRA 串联：checkpoint → lora1 → lora2 → KSampler。"""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "8": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": -clip_skip}},
        "10": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0], "clip": ["8", 0],
                "lora_name": lora1, "strength_model": strength, "strength_clip": strength,
            },
        },
        "11": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["10", 0], "clip": ["10", 1],
                "lora_name": lora2, "strength_model": strength, "strength_clip": strength,
            },
        },
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 1], "text": positive}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 1], "text": negative}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["11", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "manju"}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config_wzn.yaml"))
    ap.add_argument("--only", choices=["single", "dual"], default=None)
    ap.add_argument("--strength", type=float, default=0.6)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    img = cfg["image"]
    pb = PromptBuilder(cfg)
    out_dir = ROOT / "work_wzn" / "dual_lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = ComfyClient()
    if not client.is_up():
        raise SystemExit("ComfyUI 未启动")

    desc = {
        "shenchi": pb.chars["沈迟"].describe(),
        "sujin": pb.chars["苏锦"].describe(),
    }

    jobs = []
    if args.only in (None, "single"):
        for tag, tpl in SINGLE:
            jobs.append((tag, tpl.format(**desc)))
    if args.only in (None, "dual"):
        for tag, tpl in DUAL:
            jobs.append((tag, tpl.format(**desc)))

    for i, (tag, body) in enumerate(jobs, 1):
        positive = f"{pb.style_prefix()}, {body}, detailed environment, high detail"
        wf = build_dual_lora_workflow(
            ckpt="Illustrious-XL-v2.0.safetensors",
            positive=positive,
            negative=pb.negative(),
            width=img["base_width"], height=img["base_height"],
            steps=img["steps"], cfg=img["cfg"],
            sampler=img["sampler"], scheduler=img["scheduler"],
            seed=88000 + i, clip_skip=img.get("clip_skip", 2),
            lora1="shenchi_v1.safetensors", lora2="sujin_v1.safetensors",
            strength=args.strength,
        )
        print(f"[{i}/{len(jobs)}] {tag} …", end="", flush=True)
        files = client.generate(wf, out_dir, tag)
        print(f" {files[0].name if files else 'FAILED'}")

    print(f"\n完成，见 {out_dir}")


if __name__ == "__main__":
    main()
