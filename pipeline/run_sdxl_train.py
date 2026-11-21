"""在 ComfyUI 的 embedded Python 里运行 sd-scripts。

embedded Python 用 python313._pth 做路径隔离，PYTHONPATH 不生效，
所以这里手动把 sd-scripts 目录插进 sys.path 再执行目标脚本。

用法：
  python pipeline/run_sdxl_train.py sdxl_train_network.py --help
"""
import os
import runpy
import sys

SD_DIR = r"D:\manju\sd-scripts-0.12.3"

if not os.path.isdir(SD_DIR):
    raise SystemExit(f"找不到 sd-scripts: {SD_DIR}")

sys.path.insert(0, SD_DIR)

if len(sys.argv) < 2:
    raise SystemExit("用法: run_sdxl_train.py <script.py> [args...]")

script = sys.argv[1]
script_path = script if os.path.isabs(script) else os.path.join(SD_DIR, script)
if not os.path.exists(script_path):
    raise SystemExit(f"找不到脚本: {script_path}")

sys.argv = [script_path] + sys.argv[2:]
runpy.run_path(script_path, run_name="__main__")
