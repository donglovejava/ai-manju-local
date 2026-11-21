"""为一集生成缩略图总览（contact sheet），用于人工质检。

用法：
    python pipeline/contact_sheet.py --ep 1
    python pipeline/contact_sheet.py --dir work_wzn/ep001/images --cols 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent


def build_sheet(images_dir: Path, out: Path, cols: int = 8, thumb_w: int = 135) -> Path:
    files = sorted(images_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"{images_dir} 里没有图片")

    # 先读一张拿宽高比
    with Image.open(files[0]) as im:
        ratio = im.height / im.width
    thumb_h = int(thumb_w * ratio)
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    for i, f in enumerate(files):
        with Image.open(f) as im:
            thumb = im.convert("RGB").resize((thumb_w, thumb_h))
        x, y = (i % cols) * thumb_w, (i // cols) * thumb_h
        sheet.paste(thumb, (x, y))
        draw.text((x + 3, y + 3), f.stem, fill=(255, 90, 90))

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config_wzn.yaml"))
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--cols", type=int, default=8)
    args = ap.parse_args()

    if args.dir:
        img_dir = Path(args.dir)
        out = img_dir.parent / "contact_sheet.png"
    elif args.ep:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        work = Path(cfg["paths"]["work"]) / f"ep{args.ep:03d}"
        img_dir = work / "images"
        out = work / "contact_sheet.png"
    else:
        raise SystemExit("需要 --ep 或 --dir")

    p = build_sheet(img_dir, out, args.cols)
    n = len(list(img_dir.glob("*.png")))
    print(f"{n} 张 → {p}")


if __name__ == "__main__":
    main()
