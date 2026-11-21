"""三种「3D 动漫」风格对比测试——找出最接近「漫画 3D 人物」的提示词。

用法：
    python pipeline/wan_style_test.py            # 三个风格全跑（约 20-30 分钟）
    python pipeline/wan_style_test.py --style toon3d
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from comfy_client import ComfyClient
from wan_test import STYLE_PRESETS, build_wan_workflow, style_prompt

ROOT = Path(__file__).resolve().parent.parent

SUBJECT = (
    "一个三十岁的中国男人，深灰色大衣，站在夜晚的城市街道上，"
    "霓虹灯倒影，安静克制的表情，镜头缓慢推进"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="all", choices=["all", *STYLE_PRESETS.keys()])
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=896)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    styles = list(STYLE_PRESETS) if args.style == "all" else [args.style]
    length = 4 * round((args.seconds * 24 - 1) / 4) + 1
    out_dir = ROOT / "work" / "wan_style"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = ComfyClient(timeout=5400)
    if not client.is_up():
        raise SystemExit("ComfyUI 未启动")

    for i, style in enumerate(styles, 1):
        pos, neg = style_prompt(style, SUBJECT)
        wf = build_wan_workflow(
            positive=pos,
            negative=neg,
            width=args.width,
            height=args.height,
            length=length,
            fps=24,
            steps=args.steps,
            cfg=5.0,
            seed=1234,
            prefix=f"wan_{style}",
        )
        print(f"\n[{i}/{len(styles)}] 风格={style}  {args.width}x{args.height} {length}帧 {args.steps}步")
        print(f"  positive: {pos[:110]}…")
        t0 = time.time()
        pid = client.queue(wf)
        entry = client.wait(pid, poll=5.0)
        files = client.download_images(entry, out_dir, style)
        print(f"  完成 {time.time()-t0:.0f}s → {files[0].name if files else 'FAILED'}")

    print(f"\n全部完成，见 {out_dir}")


if __name__ == "__main__":
    main()
