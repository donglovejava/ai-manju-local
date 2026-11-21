"""角色设定图：每个主要角色出多个 seed，人工挑一张锁脸基准图。

用法：
    python pipeline/character_test.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

from comfy_client import ComfyClient, build_sdxl_workflow
from prompt_builder import PromptBuilder

ROOT = Path(__file__).resolve().parent.parent

# 中性站姿、正面、纯色背景——设定图不是剧照，要的是辨识度。
# 注意：绝对不要写 "character sheet"，Illustrious 系底模会理解成多视图拼图，极易跑出 NSFW。
POSES = {
    "陈屿": (
        "1boy, solo, only one person in frame, full body, standing straight facing camera, "
        "arms relaxed at sides, neutral calm expression, looking at viewer, "
        "wearing dark charcoal wool overcoat, black turtleneck sweater, grey trousers, black leather shoes, "
        "plain light grey studio background, even soft lighting"
    ),
    "林晚": (
        "1girl, solo, only one person in frame, full body, standing straight facing camera, "
        "arms relaxed at sides, quiet calm expression, looking at viewer, "
        "wearing grey knit sweater with worn cuffs, dark straight trousers, flat shoes, "
        "plain light grey studio background, even soft lighting"
    ),
}

SEEDS = (424242, 777001)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    img = cfg["image"]
    pb = PromptBuilder(cfg)
    out_dir = ROOT / "assets" / "chars" / "refs"
    client = ComfyClient()
    if not client.is_up():
        raise SystemExit("ComfyUI 未启动")

    for name, pose in POSES.items():
        card = pb.chars.get(name)
        if not card:
            print(f"跳过 {name}：无角色卡")
            continue
        positive = f"{pb.style_prefix()}, {card.describe()}, {pose}, high quality, masterpiece"
        for seed in SEEDS:
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
            )
            print(f"设定图: {name} seed={seed} …", end="", flush=True)
            files = client.generate(wf, out_dir, f"{name}_s{seed}")
            print(f" {files[0].name if files else 'FAILED'}")
            (out_dir / f"{name}_s{seed}.txt").write_text(positive, encoding="utf-8")

    print(f"\n完成，见 {out_dir}")


if __name__ == "__main__":
    main()
