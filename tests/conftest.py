import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "research"))

# 让测试不依赖 shell 的 PATH 设置：把项目自带的 ffmpeg 目录挂进去
_FFMPEG_BIN = Path(r"D:\manju\bin")
if _FFMPEG_BIN.exists():
    os.environ["PATH"] = str(_FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")
