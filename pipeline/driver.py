"""逐集驱动：一集一进程 + 看门狗 + 集间重启 ComfyUI。

为什么不用 batch.py：
    实测连续跑会在三种地方卡死——ComfyUI GPU 死锁（HTTP 仍 200）、
    GBK 编码崩溃、batch 与子进程整条链 CPU=0 僵死。三点各不相同，
    属于环境性不稳定，靠修单个 bug 治不好。

本脚本的对策：
    1. 每一集起独立子进程，崩了不污染下一集
    2. 看门狗盯 work 目录的文件数，STALL_SEC 秒没有任何新增就判定卡死，
       直接 kill 进程树（不傻等 40 分钟超时）
    3. 每集结束无论成败，强制重启 ComfyUI，避免状态累积
    4. 单集最多重试 MAX_RETRY 次

用法：
    python pipeline/driver.py                 # 跑全部
    python pipeline/driver.py --from 6 --to 20
    python pipeline/driver.py --ep 6 --force
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
COMFY_DIR = Path(r"D:\manju\ComfyUI_windows_portable")
COMFY_URL = "http://127.0.0.1:8188"

STALL_SEC = 360        # 6 分钟没有任何新文件 → 判定卡死
MAX_RETRY = 3          # 单集最多重试次数
EP_TIMEOUT = 50 * 60   # 单集硬上限


# ---------- ComfyUI 控制 ----------
def _comfy_ok(timeout: float = 5) -> bool:
    try:
        import requests

        return requests.get(f"{COMFY_URL}/system_stats", timeout=timeout).ok
    except Exception:
        return False


def _comfy_pid() -> int | None:
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=20
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if ":8188" in line and "LISTENING" in line.upper():
            try:
                return int(line.split()[-1])
            except Exception:
                return None
    return None


def restart_comfy() -> bool:
    """强制重启 ComfyUI。HTTP 200 不代表 GPU 没死锁，所以不看状态直接重来。"""
    pid = _comfy_pid()
    if pid:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        time.sleep(8)
    py = COMFY_DIR / "python_embeded" / "python.exe"
    if not py.exists():
        print(f"  找不到 {py}")
        return False
    subprocess.Popen(
        [str(py), "-s", "ComfyUI/main.py", "--port", "8188", "--preview-method", "none"],
        cwd=str(COMFY_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        time.sleep(2)
        if _comfy_ok():
            return True
    return False


# ---------- 看门狗 ----------
def _count_files(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.rglob("*") if p.is_file())


def run_one(ep: int, shots_file: Path, work_dir: Path, log_path: Path) -> str:
    """跑一集，返回 ok / stalled / failed。看门狗盯着文件数变化。"""
    cmd = [
        PY, str(ROOT / "pipeline" / "run_episode.py"),
        "--config", str(ROOT / "config_wzn.yaml"),
        "--shots-file", str(shots_file),
        "--episode", str(ep),
        "--max-shots", "70",
        "--lora", "shenchi_v3.safetensors",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"\n\n===== EP{ep:03d} 开始 {time.strftime('%H:%M:%S')} =====\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT,
        )
        start = time.time()
        last_count = _count_files(work_dir)
        last_change = time.time()

        while proc.poll() is None:
            time.sleep(20)
            n = _count_files(work_dir)
            if n != last_count:
                last_count = n
                last_change = time.time()
            stalled = time.time() - last_change > STALL_SEC
            overtime = time.time() - start > EP_TIMEOUT
            if stalled or overtime:
                why = "卡死（长时间无产出）" if stalled else "超时（50 分钟）"
                lf.write(f"\n===== EP{ep:03d} {why}，强制终止 =====\n")
                lf.flush()
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True
                )
                proc.wait(timeout=30)
                return "stalled" if stalled else "failed"

    return "ok" if proc.returncode == 0 else "failed"


# ---------- 主流程 ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-ep", type=int, default=70)
    ap.add_argument("--from", dest="ep_from", type=int, default=None)
    ap.add_argument("--to", dest="ep_to", type=int, default=None)
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config_wzn.yaml").read_text(encoding="utf-8"))
    work = Path(cfg["paths"]["work"])
    out_dir = Path(cfg["paths"]["output"])
    eps_dir = work / "eps"
    prog_path = work / "driver_progress.json"
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    sys.path.insert(0, str(ROOT / "pipeline"))
    from batch import build_plan

    plan = build_plan(cfg, args.per_ep)
    targets = plan
    if args.ep:
        targets = [p for p in plan if p["ep"] == args.ep]
    else:
        if args.ep_from:
            targets = [t for t in targets if t["ep"] >= args.ep_from]
        if args.ep_to:
            targets = [t for t in targets if t["ep"] <= args.ep_to]

    progress: dict = {}
    if prog_path.exists():
        progress = json.loads(prog_path.read_text(encoding="utf-8"))

    def save() -> None:
        prog_path.write_text(
            json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"共 {len(plan)} 集，本次处理 {len(targets)} 集")
    ok_cnt = skip_cnt = 0

    for item in targets:
        ep = item["ep"]
        key = str(ep)
        rec = progress.get(key, {})
        if rec.get("status") == "done" and not args.force:
            skip_cnt += 1
            continue

        # 每集开跑前重启 ComfyUI，杜绝状态累积
        print(f"\n=== EP{ep:03d} === 重启 ComfyUI …", flush=True)
        if not restart_comfy():
            print("  ComfyUI 起不来，跳过")
            progress[key] = {"status": "failed", "error": "ComfyUI 无法启动"}
            save()
            continue

        shots_file = eps_dir / f"ep{ep:03d}_shots.json"
        shots_file.write_text(
            json.dumps(
                {"chapter": item["chapters"][0], "title": item["title"],
                 "shots": item["shots"], "ep": ep},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        work_dir = work / f"ep{ep:03d}"
        log_path = log_dir / f"ep{ep:03d}.log"

        result = "failed"
        for attempt in range(1, MAX_RETRY + 1):
            t0 = time.time()
            print(f"  第 {attempt} 次尝试 …", end="", flush=True)
            result = run_one(ep, shots_file, work_dir, log_path)
            mins = (time.time() - t0) / 60
            print(f" {result}（{mins:.1f} 分钟）", flush=True)
            if result == "ok":
                break
            if attempt < MAX_RETRY:
                print("  重启 ComfyUI 后重试", flush=True)
                restart_comfy()

        if result == "ok":
            progress[key] = {
                "status": "done", "shots": len(item["shots"]),
                "chapters": item["chapters"],
                "path": str(out_dir / f"ep{ep:03d}.mp4"),
            }
            ok_cnt += 1
        else:
            progress[key] = {"status": "failed", "error": result}
        save()

    done = sum(1 for v in progress.values() if v.get("status") == "done")
    print(f"\n本次成功 {ok_cnt} 集，跳过 {skip_cnt} 集，累计完成 {done} 集")


if __name__ == "__main__":
    main()
