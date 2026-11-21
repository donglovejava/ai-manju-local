"""单镜头 IP-Adapter 出图测试（快速验证锁脸是否生效）。"""
from pathlib import Path

import yaml

from comfy_client import ComfyClient, build_sdxl_workflow
from prompt_builder import PromptBuilder

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    img = cfg["image"]
    pb = PromptBuilder(cfg)
    client = ComfyClient()
    if not client.is_up():
        raise SystemExit("ComfyUI 未启动")

    out = ROOT / "work" / "_ipa_test"
    shot = {
        "shot_type": "close",
        "characters": [{"name": "陈屿", "expression": "calm", "pose": "sitting at a desk"}],
        "action": "holding a smartphone, screen lighting up",
        "setting": "open plan office, desk with laptop, cool fluorescent light",
        "props": ["smartphone"],
        "mood": "平静",
    }
    pos, neg = pb.build(shot)
    for w in (0.45, 0.65):
        wf = build_sdxl_workflow(
            ckpt_name="Illustrious-XL-v2.0.safetensors",
            positive=pos,
            negative=neg,
            width=img["base_width"],
            height=img["base_height"],
            steps=img["steps"],
            cfg=img["cfg"],
            sampler=img["sampler"],
            scheduler=img["scheduler"],
            clip_skip=img.get("clip_skip", 2),
            seed=246810,
            ipadapter_ref="ref_chenyu_head.png",
            ipadapter_weight=w,
        )
        print(f"IP-Adapter weight={w} …", end="", flush=True)
        files = client.generate(wf, out, f"ipa_w{w}")
        print(f" {files[0].name if files else 'FAILED'}")


if __name__ == "__main__":
    main()
