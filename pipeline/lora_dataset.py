"""为角色 LoRA 批量生成训练素材。

目标：同一角色 × 多角度 × 多表情 × 多场景，输出 60 张候选图供人工筛选。
训练素材的批量 ≠ 正片：这里刻意变化角度和构图，避免 LoRA 只会画一种姿势。

用法：
    python pipeline/lora_dataset.py --char 沈迟 --config config_wzn.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from comfy_client import ComfyClient, build_sdxl_workflow
from prompt_builder import PromptBuilder

ROOT = Path(__file__).resolve().parent.parent

# 12 种构图/姿态 × 5 个 seed = 60 张
VARIATIONS = [
    ("front", "facing camera directly, neutral calm expression, looking at viewer, plain background"),
    ("three_quarter", "three quarter view, head turned slightly to the left, calm expression"),
    ("profile", "side profile view, looking to the right, quiet expression"),
    ("closeup", "close-up of face and shoulders, calm eyes, subtle expression"),
    ("upper_body", "upper body shot, arms relaxed, slight head tilt"),
    ("sitting_desk", "sitting at a wooden desk, one hand resting on the surface, reading"),
    ("window", "standing by a large window, looking outside, soft daylight on face"),
    ("snow", "walking in falling snow, coat collar turned up, breath vapor"),
    ("phone", "holding a smartphone in both hands, looking down at the screen"),
    ("lying", "lying on a bed, eyes open, looking upward, soft dim light"),
    ("back_view", "back view, head turned back over one shoulder, looking at viewer"),
    ("smile", "faint restrained smile, warm soft lighting, three quarter view"),
]

SEEDS = (1101, 2202, 3303, 4404, 5505)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config_wzn.yaml"))
    ap.add_argument("--char", default="沈迟")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--only", default=None, help="只跑某个 variation 名")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    img = cfg["image"]
    pb = PromptBuilder(cfg)
    card = pb.chars.get(args.char)
    if not card:
        raise SystemExit(f"角色卡里没有 {args.char}")

    out_dir = ROOT / "assets" / "lora" / args.char
    out_dir.mkdir(parents=True, exist_ok=True)

    client = ComfyClient()
    if not client.is_up():
        raise SystemExit("ComfyUI 未启动")

    variations = [v for v in VARIATIONS if args.only in (None, v[0])]
    total = len(variations) * len(SEEDS)
    n = 0
    t0 = time.time()

    for vname, vdesc in variations:
        # 训练素材要「角色占主导」，风格词适度但不要压过人物
        positive = (
            f"{pb.style_prefix()}, {card.describe()}, {vdesc}, "
            "single character, full attention on the face, sharp focus, high detail"
        )
        negative = pb.negative()
        for seed in SEEDS:
            n += 1
            dst = out_dir / f"{args.char}_{vname}_s{seed}.png"
            if dst.exists():
                print(f"  [{n}/{total}] 跳过 {dst.name}")
                continue
            wf = build_sdxl_workflow(
                ckpt_name="Illustrious-XL-v2.0.safetensors",
                positive=positive,
                negative=negative,
                width=img["base_width"],
                height=img["base_height"],
                steps=args.steps,
                cfg=img["cfg"],
                sampler=img["sampler"],
                scheduler=img["scheduler"],
                clip_skip=img.get("clip_skip", 2),
                seed=seed,
            )
            print(f"  [{n}/{total}] {vname} seed={seed} …", end="", flush=True)
            try:
                files = client.generate(wf, out_dir, f"tmp_{vname}_{seed}")
                if files:
                    files[0].rename(dst)
                    for extra in files[1:]:
                        extra.unlink(missing_ok=True)
                    print(" ok")
                else:
                    print(" 无输出")
            except Exception as e:  # noqa: BLE001
                print(f" 失败: {e}")

    print(f"\n完成 {n} 张，耗时 {(time.time()-t0)/60:.1f} 分钟 → {out_dir}")


if __name__ == "__main__":
    main()
