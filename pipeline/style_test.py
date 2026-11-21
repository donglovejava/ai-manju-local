"""s4 画风对比测试：同一个场景，四种风格配置各出一张，供人工选型。

用法（ComfyUI 需已启动）：
    python pipeline/style_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from comfy_client import ComfyClient, build_sdxl_workflow

ROOT = Path(__file__).resolve().parent.parent

# 同一个场景，只改风格前缀，变量控制
SCENE = (
    "a 30 year old East Asian man with short black hair, sitting alone at a crowded "
    "Chinese restaurant table at night, holding a white envelope under the table, "
    "melancholic restrained expression, city lights through window"
)

STYLES = {
    "korean_manhwa": (
        "korean manhwa style, webtoon illustration, clean confident lineart, "
        "soft cel shading, muted elegant color palette, cinematic lighting, "
        "detailed eyes, high quality, masterpiece, best quality"
    ),
    "japanese_anime": (
        "japanese anime style, cel shading, crisp outlines, vibrant but soft colors, "
        "makoto shinkai lighting, volumetric light, detailed background, "
        "high quality, masterpiece, best quality"
    ),
    "chinese_donghua": (
        "chinese donghua style, semi realistic anime, rich detailed rendering, "
        "dramatic rim lighting, textured brushwork, cinematic composition, "
        "high quality, masterpiece, best quality"
    ),
    "korean_manhwa_clean": (
        "korean webtoon style, flat clean colors, minimal shading, thin precise lineart, "
        "modern urban aesthetic, limited palette of grey and warm accent, "
        "high quality, masterpiece, best quality"
    ),
}

NEGATIVE = (
    "lowres, bad anatomy, bad hands, extra fingers, missing fingers, fused fingers, "
    "worst quality, low quality, jpeg artifacts, watermark, signature, text, logo, "
    "blurry, deformed, disfigured, extra limbs, multiple views, collage, comic panel, "
    "speech bubble, border, frame, chibi, 3d render, photorealistic"
)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    img = cfg["image"]
    out_dir = ROOT / "work" / "style_test"
    client = ComfyClient()
    if not client.is_up():
        raise SystemExit("ComfyUI 未启动")

    for name, prefix in STYLES.items():
        wf = build_sdxl_workflow(
            ckpt_name="Illustrious-XL-v2.0.safetensors",
            positive=f"{prefix}, {SCENE}",
            negative=NEGATIVE,
            width=img["base_width"],
            height=img["base_height"],
            steps=img["steps"],
            cfg=img["cfg"],
            sampler=img["sampler"],
            scheduler=img["scheduler"],
            clip_skip=img.get("clip_skip", 2),
            seed=1234567,  # 固定种子，只有风格变量不同
        )
        print(f"出图: {name} …", end="", flush=True)
        files = client.generate(wf, out_dir, name)
        print(f" {files[0].name if files else 'FAILED'}")
        (out_dir / f"{name}.txt").write_text(f"{prefix}, {SCENE}", encoding="utf-8")

    print(f"\n完成，见 {out_dir}")


if __name__ == "__main__":
    main()
