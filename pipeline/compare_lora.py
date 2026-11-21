"""对比同一角色的两版 LoRA（v1 vs v2），同 seed 同 prompt。

用法：
    python pipeline/compare_lora.py --char 苏锦 --a sujin_v1.safetensors --b sujin_v2.safetensors
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from comfy_client import ComfyClient, build_sdxl_workflow
from prompt_builder import PromptBuilder

ROOT = Path(__file__).resolve().parent.parent

PROBES = [
    ("closeup", "close-up of face and shoulders, looking at viewer, calm expression"),
    ("three_quarter", "three quarter view, standing in a school corridor, soft daylight"),
    ("full_body", "full body, standing by a large window, looking outside"),
]

SEED = 91001


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config_wzn.yaml"))
    ap.add_argument("--char", default="苏锦")
    ap.add_argument("--trigger", default="sujin")
    ap.add_argument("--a", default="sujin_v1.safetensors")
    ap.add_argument("--b", default="sujin_v2.safetensors")
    ap.add_argument("--strength", type=float, default=0.7)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    img = cfg["image"]
    pb = PromptBuilder(cfg)
    card = pb.chars[args.char]
    out_dir = ROOT / "work_wzn" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = ComfyClient()
    if not client.is_up():
        raise SystemExit("ComfyUI 未启动")

    base = f"{pb.style_prefix()}, {card.describe()}"
    for name, desc in PROBES:
        for tag, lora in (("v1", args.a), ("v2", args.b)):
            positive = f"{args.trigger}, {base}, {desc}, single character, high detail"
            wf = build_sdxl_workflow(
                ckpt_name="Illustrious-XL-v2.0.safetensors",
                positive=positive,
                negative=pb.negative(),
                width=img["base_width"], height=img["base_height"],
                steps=img["steps"], cfg=img["cfg"],
                sampler=img["sampler"], scheduler=img["scheduler"],
                clip_skip=img.get("clip_skip", 2), seed=SEED,
                lora_name=lora, lora_strength=args.strength,
            )
            prefix = f"{name}_{tag}"
            print(f"{prefix} …", end="", flush=True)
            files = client.generate(wf, out_dir, prefix)
            print(f" {files[0].name if files else 'FAILED'}")

    print(f"\n完成，见 {out_dir}")


if __name__ == "__main__":
    main()
