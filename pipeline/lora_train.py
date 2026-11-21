"""用 ComfyUI 内置的 TrainLoraNode 训练角色 LoRA。

不需要 kohya_ss / sd-scripts；节点链：
  CheckpointLoaderSimple → CLIPTextEncode(触发词)
  LoadImageDataSetFromFolder → VAEEncode
  TrainLoraNode(model, latents, positive) → SaveLoRA

前置：训练图必须放在 ComfyUI/input/<folder>/ 下（节点只扫描 input 子目录）。

用法：
  python pipeline/lora_train.py --folder shenchi_lora --trigger "shenchi" --steps 1200
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from comfy_client import ComfyClient

ROOT = Path(__file__).resolve().parent.parent
COMFY_INPUT = Path(r"D:\manju\ComfyUI_windows_portable\ComfyUI\input")


def build_train_workflow(
    *,
    folder: str,
    ckpt: str = "Illustrious-XL-v2.0.safetensors",
    trigger: str = "shenchi",
    steps: int = 1200,
    rank: int = 16,
    lr: float = 1e-4,
    batch_size: int = 1,
    grad_accum: int = 1,
    out_name: str = "shenchi_v1",
    seed: int = 42,
) -> dict:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": f"{trigger}, a young Chinese man, short black hair, dark navy wool coat, grey knit sweater",
            },
        },
        "3": {
            "class_type": "LoadImageDataSetFromFolder",
            "inputs": {"folder": folder},
        },
        "4": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["3", 0], "vae": ["1", 2]},
        },
        "5": {
            "class_type": "TrainLoraNode",
            "inputs": {
                "model": ["1", 0],
                "latents": ["4", 0],
                "positive": ["2", 0],
                "batch_size": batch_size,
                "grad_accumulation_steps": grad_accum,
                "steps": steps,
                "learning_rate": lr,
                "rank": rank,
                "optimizer": "AdamW",
                "loss_function": "MSE",
                "seed": seed,
                "training_dtype": "bf16",
                "lora_dtype": "bf16",
                "quantized_backward": False,
                "algorithm": "LoRA",
                "gradient_checkpointing": True,
                "checkpoint_depth": 1,
                "offloading": True,
                "existing_lora": "[None]",
                "bucket_mode": False,
                "bypass_mode": False,
            },
        },
        "6": {
            "class_type": "SaveLoRA",
            "inputs": {
                "lora": ["5", 0],
                "prefix": f"loras/{out_name}",
            },
        },
    }


def prepare_dataset(src_dir: Path, folder: str) -> int:
    """把训练图复制到 ComfyUI/input/<folder>/，返回张数。"""
    dst = COMFY_INPUT / folder
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src_dir.glob("*.png")):
        target = dst / f.name
        if not target.exists():
            target.write_bytes(f.read_bytes())
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="shenchi_lora")
    ap.add_argument("--src", default=str(ROOT / "assets" / "lora" / "沈迟" / "picked"))
    ap.add_argument("--trigger", default="shenchi")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default="shenchi_v1")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"训练集目录不存在: {src}")

    n = prepare_dataset(src, args.folder)
    print(f"训练集 {n} 张 → {COMFY_INPUT / args.folder}")

    client = ComfyClient(timeout=10800)  # 训练可能 1-3 小时
    if not client.is_up():
        raise SystemExit("ComfyUI 未启动")

    wf = build_train_workflow(
        folder=args.folder, trigger=args.trigger, steps=args.steps,
        rank=args.rank, lr=args.lr, batch_size=args.batch, out_name=args.out,
    )
    print(f"提交训练：{n} 张 / {args.steps} 步 / rank {args.rank} / lr {args.lr}")
    t0 = time.time()
    pid = client.queue(wf)
    print(f"prompt_id={pid}，等待训练完成…")
    try:
        entry = client.wait(pid, poll=10.0)
        print(f"训练完成，耗时 {(time.time()-t0)/60:.1f} 分钟")
        for node_out in entry.get("outputs", {}).values():
            print("输出:", node_out)
    except Exception as e:  # noqa: BLE001
        print(f"训练失败: {e}")


if __name__ == "__main__":
    main()
