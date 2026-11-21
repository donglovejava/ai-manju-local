"""ComfyUI HTTP API 客户端。

只做三件事：提交 SDXL 文生图 workflow、轮询结果、把图下载到本地。
不引入 comfy 的 Python 依赖，纯 requests，方便在系统 Python 里跑。
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import requests

DEFAULT_SERVER = "http://127.0.0.1:8188"


class ComfyClient:
    def __init__(self, server: str = DEFAULT_SERVER, timeout: int = 600):
        self.server = server.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self.timeout = timeout

    # ---------- 基础 ----------
    def is_up(self) -> bool:
        try:
            return requests.get(f"{self.server}/system_stats", timeout=3).ok
        except requests.RequestException:
            return False

    def free_memory(self) -> None:
        """释放显存和内存缓存的模型（8GB 卡在出图/配音之间切换时必用）。"""
        try:
            requests.post(
                f"{self.server}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=10,
            )
        except requests.RequestException:
            pass

    # ---------- 提交 ----------
    def queue(self, workflow: dict[str, Any]) -> str:
        resp = requests.post(
            f"{self.server}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            timeout=30,
        )
        if not resp.ok:
            # ComfyUI 的 400 会把具体校验失败写在 body 里，必须打出来
            raise RuntimeError(f"提交失败 HTTP {resp.status_code}: {resp.text[:1200]}")
        body = resp.json()
        if body.get("node_errors"):
            raise RuntimeError(f"workflow 校验失败: {json.dumps(body['node_errors'], ensure_ascii=False)}")
        return body["prompt_id"]

    # ---------- 等待 ----------
    def wait(self, prompt_id: str, poll: float = 1.5) -> dict[str, Any]:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            hist = requests.get(f"{self.server}/history/{prompt_id}", timeout=10).json()
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI 执行出错: {json.dumps(status, ensure_ascii=False)[:800]}")
                if status.get("completed", True):
                    return entry
            time.sleep(poll)
        raise TimeoutError(f"等 ComfyUI 出图超时（{self.timeout}s）: {prompt_id}")

    # ---------- 取图 ----------
    def download_images(self, entry: dict[str, Any], out_dir: Path, prefix: str) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        idx = 0
        for node_out in entry.get("outputs", {}).values():
            for img in node_out.get("images", []):
                idx += 1
                params = {
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                }
                r = requests.get(f"{self.server}/view", params=params, timeout=60)
                r.raise_for_status()
                suffix = Path(img["filename"]).suffix or ".png"
                dst = out_dir / f"{prefix}_{idx:03d}{suffix}"
                dst.write_bytes(r.content)
                saved.append(dst)
        return saved

    # ---------- 一站式 ----------
    def generate(self, workflow: dict[str, Any], out_dir: Path, prefix: str) -> list[Path]:
        pid = self.queue(workflow)
        entry = self.wait(pid)
        return self.download_images(entry, out_dir, prefix)


def build_sdxl_workflow(
    *,
    ckpt_name: str,
    positive: str,
    negative: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    cfg: float = 5.5,
    sampler: str = "dpmpp_2m",
    scheduler: str = "karras",
    seed: int = -1,
    batch_size: int = 1,
    clip_skip: int = 2,
    lora_name: str | None = None,
    lora_strength: float = 0.8,
    ipadapter_ref: str | None = None,
    ipadapter_weight: float = 0.8,
    ipadapter_preset: str = "PLUS (high strength)",
) -> dict[str, Any]:
    """构建 SDXL 文生图的 API-format workflow。

    节点编号固定，便于调试时对照：
      1 CheckpointLoaderSimple -> 2 CLIPTextEncode(正) / 3 CLIPTextEncode(负)
      4 EmptyLatentImage -> 5 KSampler -> 6 VAEDecode -> 7 SaveImage
      10 LoraLoader（可选）
    """
    if seed is None or seed < 0:
        seed = int(time.time() * 1000) % (2**32)

    model_ref: list[Any] = ["1", 0]
    clip_ref: list[Any] = ["1", 1]

    wf: dict[str, Any] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": clip_ref},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": clip_ref},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": batch_size},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": model_ref,
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "manju"},
        },
        "8": {
            "class_type": "CLIPSetLastLayer",
            "inputs": {"clip": ["1", 1], "stop_at_clip_layer": -clip_skip},
        },
    }
    # clip_skip 生效：把正向/负向接到 CLIPSetLastLayer 的输出
    wf["2"]["inputs"]["clip"] = ["8", 0]
    wf["3"]["inputs"]["clip"] = ["8", 0]

    if lora_name:
        wf["10"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["8", 0],
                "lora_name": lora_name,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
            },
        }
        wf["5"]["inputs"]["model"] = ["10", 0]
        wf["2"]["inputs"]["clip"] = ["10", 1]
        wf["3"]["inputs"]["clip"] = ["10", 1]

    # IP-Adapter：接在 LoRA 之后，用角色设定图锁脸（这是跨镜头一致性的关键）
    if ipadapter_ref:
        wf["20"] = {
            "class_type": "IPAdapterUnifiedLoader",
            "inputs": {"model": wf["5"]["inputs"]["model"], "preset": ipadapter_preset},
        }
        wf["30"] = {"class_type": "LoadImage", "inputs": {"image": ipadapter_ref}}
        wf["21"] = {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["20", 0],
                "ipadapter": ["20", 1],
                "image": ["30", 0],
                "weight": ipadapter_weight,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 1.0,
                "embeds_scaling": "V only",
            },
        }
        wf["5"]["inputs"]["model"] = ["21", 0]

    return wf


if __name__ == "__main__":
    c = ComfyClient()
    print("ComfyUI up:", c.is_up())
