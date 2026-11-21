"""角色 LoRA 效果验证：同一触发词、不同 seed，看脸是否稳定。

用法：
    python pipeline/lora_test.py --lora shenchi_v1.safetensors --trigger shenchi
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from comfy_client import ComfyClient, build_sdxl_workflow
from prompt_builder import PromptBuilder

ROOT = Path(__file__).resolve().parent.parent

# 与训练集不同的场景/姿态，检验泛化而非单纯记忆
PROBES = [
    ("closeup", "close-up of face and shoulders, looking at viewer, calm expression"),
    ("street", "standing on a city street at night, holding a phone, three quarter view"),
    ("desk", "sitting at a wooden desk, reading a book, warm desk lamp"),
    ("rain", "standing in light rain, coat collar up, looking away"),
]

SEEDS = (70001, 70002)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config_wzn.yaml"))
    ap.add_argument("--lora", default="shenchi_v1.safetensors")
    ap.add_argument("--trigger", default="shenchi")
    ap.add_argument("--strength", type=float, default=0.85)
    ap.add_argument("--compare-none", action="store_true", help="同时出一组不带 LoRA 的对照")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    img = cfg["image"]
    pb = PromptBuilder(cfg)
    out_dir = ROOT / "work_wzn" / "lora_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = ComfyClient()
    if not client.is_up():
        raise SystemExit("ComfyUI 未启动")

    card = pb.chars["沈迟"]
    base = f"{pb.style_prefix()}, {card.describe()}"

    for name, desc in PROBES:
        for seed in SEEDS:
            for use_lora in ([True, False] if args.compare_none else [True]):
                positive = f"{args.trigger}, {base}, {desc}, single character, high detail"
                wf = build_sdxl_workflow(
                    ckpt_name="Illustrious-XL-v2.0.safetensors",
                    positive=positive,
                    negative=pb.negative(),
                    width=img["base_width"],
                    height=img["base_height"],
                    steps=img["steps"],
                    cfg=img["cfg"],
                    sampler=img["sampler"],
                    scheduler=img["scheduler"],
                    clip_skip=img.get("clip_skip", 2),
                    seed=seed,
                    lora_name=args.lora if use_lora else None,
                    lora_strength=args.strength,
                )
                tag = "lora" if use_lora else "nolora"
                prefix = f"{name}_s{seed}_{tag}"
                print(f"{prefix} …", end="", flush=True)
                files = client.generate(wf, out_dir, prefix)
                print(f" {files[0].name if files else 'FAILED'}")

    print(f"\n完成，见 {out_dir}")


if __name__ == "__main__":
    main()
