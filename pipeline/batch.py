"""批量生产总控：分镜 → 切集 → 逐集出片，支持断点续跑。

用法：
    # 1. 先生成所有章节的分镜（可反复执行，已完成的会跳过）
    python pipeline/batch.py --stage storyboard

    # 2. 出片（默认每集 47 镜 ≈ 1.7 分钟）
    python pipeline/batch.py --stage episodes --per-ep 47
    python pipeline/batch.py --stage episodes --from 1 --to 5      # 只跑前 5 集
    python pipeline/batch.py --stage episodes --ep 3 --force       # 强制重跑第 3 集

    # 3. 看进度
    python pipeline/batch.py --stage status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Windows 控制台默认 GBK。日志里混进非法字符时 print 会抛 UnicodeEncodeError，
# 直接把整个 batch 进程干掉（实测凌晨 EP009 就是这么崩的）。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def load_progress(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"storyboard": {"done": [], "failed": {}}, "episodes": {}}


def save_progress(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- 阶段 1：分镜 ----------
def stage_storyboard(cfg: dict, progress: dict, prog_path: Path, only: list[int] | None = None) -> None:
    work = Path(cfg["paths"]["work"])
    data = json.loads((work / "scenes.json").read_text(encoding="utf-8"))
    chapters = [c["index"] for c in data["chapters"]]
    if only:
        chapters = [c for c in chapters if c in only]

    done = set(progress["storyboard"]["done"])
    todo = [c for c in chapters if c not in done and not (work / f"shots_ch{c:03d}.json").exists()]
    print(f"分镜：共 {len(chapters)} 章，已完成 {len(done)}，本次待生成 {len(todo)}")

    for i, ch in enumerate(todo, 1):
        t0 = time.time()
        cmd = [
            PY, str(ROOT / "pipeline" / "storyboard.py"),
            "--config", str(ROOT / "config_wzn.yaml"),
            "--chapter", str(ch), "--llm",
        ]
        print(f"[{i}/{len(todo)}] 第{ch}章 分镜 …", end="", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0:
            progress["storyboard"]["done"].append(ch)
            progress["storyboard"]["failed"].pop(str(ch), None)
            print(f" ok ({(time.time()-t0)/60:.1f}分)")
        else:
            progress["storyboard"]["failed"][str(ch)] = r.stderr[-400:]
            print(f" 失败: {r.stderr[-200:]}")
        save_progress(prog_path, progress)


# ---------- 阶段 2：切集 ----------
def build_plan(cfg: dict, per_ep: int) -> list[dict]:
    """按镜头数跨章切集，返回 [{ep, shots, title, chapters}]。"""
    work = Path(cfg["paths"]["work"])
    files = sorted(work.glob("shots_ch*.json"))
    if not files:
        raise SystemExit("没有分镜文件，先跑 --stage storyboard")

    flat: list[tuple[int, str, dict]] = []  # (chapter, title, shot)
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d["shots"]:
            flat.append((d["chapter"], d.get("title") or "", s))

    plan: list[dict] = []
    buf: list[tuple[int, str, dict]] = []
    for item in flat:
        buf.append(item)
        if len(buf) >= per_ep:
            plan.append(_make_ep(len(plan) + 1, buf))
            buf = []
    if buf:
        plan.append(_make_ep(len(plan) + 1, buf))
    return plan


def _make_ep(ep: int, buf: list[tuple[int, str, dict]]) -> dict:
    chapters = sorted({c for c, _, _ in buf})
    title = buf[0][1]
    # 必须重编 id：每一章的 shots 都从 001 开始，跨章合并后 id 会撞车。
    # 撞车的后果是图片按 id 命名互相覆盖，一集 70 镜实际只存下 40 多张，画面循环重复。
    shots: list[dict] = []
    for i, (_, _, s) in enumerate(buf, 1):
        ns = dict(s)
        ns["id"] = f"{i:03d}"
        shots.append(ns)
    return {
        "ep": ep,
        "shots": shots,
        "chapters": chapters,
        "title": title,
    }


# ---------- 阶段 3：出片 ----------
COMFY_DIR = Path(r"D:\manju\ComfyUI_windows_portable")
COMFY_URL = "http://127.0.0.1:8188"


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


def _start_comfy() -> bool:
    py = COMFY_DIR / "python_embeded" / "python.exe"
    if not py.exists():
        print(f"找不到 {py}，无法自动重启")
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
            print("ComfyUI 已恢复")
            return True
    print("ComfyUI 重启失败")
    return False


def force_restart_comfyui() -> bool:
    """无视 HTTP 状态强制重启。

    HTTP 200 不代表 GPU 没死锁——实测凌晨 1:04 后 ComfyUI 一直返回 200、
    队列为空，但连续 6 小时一张图都出不来，只能靠 taskkill 恢复。
    """
    pid = _comfy_pid()
    if pid:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        time.sleep(8)
    return _start_comfy()


def ensure_comfyui() -> bool:
    """每集开始前探活：HTTP 不通才重启（正常情况不该无谓重启）。"""
    if _comfy_ok():
        return True
    print("ComfyUI 无响应，尝试重启 …")
    return force_restart_comfyui()


def stage_episodes(
    cfg: dict, progress: dict, prog_path: Path,
    per_ep: int, ep_from: int | None, ep_to: int | None,
    only_ep: int | None, force: bool,
) -> None:
    work = Path(cfg["paths"]["work"])
    plan = build_plan(cfg, per_ep)
    eps_dir = work / "eps"
    eps_dir.mkdir(parents=True, exist_ok=True)

    targets = plan
    if only_ep:
        targets = [p for p in plan if p["ep"] == only_ep]
    else:
        if ep_from:
            targets = [p for p in targets if p["ep"] >= ep_from]
        if ep_to:
            targets = [t for t in targets if t["ep"] <= ep_to]

    print(f"共 {len(plan)} 集（每集 {per_ep} 镜），本次处理 {len(targets)} 集")

    for item in targets:
        ep = item["ep"]
        key = str(ep)
        rec = progress["episodes"].get(key, {})
        if rec.get("status") == "done" and not force:
            print(f"EP{ep:03d} 已完成，跳过")
            continue

        # 每集开始前探活：ComfyUI 长跑会偶发无响应，挂了就自动重启
        if not ensure_comfyui():
            print(f"EP{ep:03d} 跳过：ComfyUI 不可用")
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

        cmd = [
            PY, str(ROOT / "pipeline" / "run_episode.py"),
            "--config", str(ROOT / "config_wzn.yaml"),
            "--shots-file", str(shots_file),
            "--episode", str(ep),
            "--max-shots", str(per_ep),
            "--lora", "shenchi_v1.safetensors",
        ]
        print(f"\n=== EP{ep:03d}  {item['title']}  章节 {item['chapters']} ===")
        t0 = time.time()
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=2400,
            )
            elapsed = (time.time() - t0) / 60
            ok = r.returncode == 0
            detail = (r.stderr or r.stdout)[-600:]
        except subprocess.TimeoutExpired:
            elapsed = (time.time() - t0) / 60
            ok = False
            detail = (
                "超时 40 分钟（ComfyUI 无响应或 GPU 死锁），已强制终止并重启 ComfyUI；"
                "已有图片会保留，重跑该集时自动跳过"
            )
            # 必须强制重启：HTTP 200 掩盖了 GPU 死锁，不重启会一直空转
            force_restart_comfyui()

        if ok:
            out = Path(cfg["paths"]["output"]) / f"ep{ep:03d}.mp4"
            progress["episodes"][key] = {
                "status": "done",
                "shots": len(item["shots"]),
                "chapters": item["chapters"],
                "minutes": round(elapsed, 1),
                "path": str(out),
            }
            print(f"EP{ep:03d} 完成，{elapsed:.1f} 分钟 → {out}")
        else:
            progress["episodes"][key] = {
                "status": "failed",
                "error": detail,
                "minutes": round(elapsed, 1),
            }
            print(f"EP{ep:03d} 失败：{detail[-300:]}")
        save_progress(prog_path, progress)


def stage_status(progress: dict, per_ep: int, cfg: dict) -> None:
    plan = build_plan(cfg, per_ep)
    eps = progress.get("episodes", {})
    done = [k for k, v in eps.items() if v.get("status") == "done"]
    failed = [k for k, v in eps.items() if v.get("status") == "failed"]
    total_min = sum(v.get("minutes", 0) for v in eps.values())
    print(f"分镜完成: {len(progress['storyboard']['done'])} 章")
    print(f"计划集数: {len(plan)}（每集 {per_ep} 镜）")
    print(f"已完成:   {len(done)} 集")
    print(f"失败:     {len(failed)} 集 {failed[:10]}")
    print(f"累计耗时: {total_min/60:.1f} 小时")
    if done:
        print(f"剩余:     {len(plan) - len(done)} 集")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config_wzn.yaml"))
    ap.add_argument("--stage", required=True, choices=["storyboard", "episodes", "status"])
    ap.add_argument("--per-ep", type=int, default=47)
    ap.add_argument("--from", dest="ep_from", type=int, default=None)
    ap.add_argument("--to", dest="ep_to", type=int, default=None)
    ap.add_argument("--ep", type=int, default=None, help="只跑某一集")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--chapters", default=None, help="只生成这些章的分镜，如 1,2,3")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    prog_path = Path(cfg["paths"]["work"]) / "batch_progress.json"
    progress = load_progress(prog_path)

    if args.stage == "storyboard":
        only = [int(x) for x in args.chapters.split(",")] if args.chapters else None
        stage_storyboard(cfg, progress, prog_path, only)
    elif args.stage == "episodes":
        stage_episodes(
            cfg, progress, prog_path, args.per_ep,
            args.ep_from, args.ep_to, args.ep, args.force,
        )
    else:
        stage_status(progress, args.per_ep, cfg)


if __name__ == "__main__":
    main()
