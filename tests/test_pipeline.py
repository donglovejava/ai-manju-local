"""流水线回归测试（pytest）。

运行：
    cd C:\\project\\manju
    export PATH="/d/manju/bin:$PATH"
    python -m pytest tests/ -q
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_config_key_fields():
    cfg = load_cfg()
    assert cfg["image"]["base_width"] == 768
    assert cfg["image"]["base_height"] == 1344
    assert "safe" in cfg["style"]["positive_prefix"]
    assert "nsfw" in cfg["style"]["negative"]
    assert cfg["style"]["preset"] == "korean_manhwa_clean"


def test_all_modules_import():
    import comfy_client  # noqa: F401
    import prompt_builder  # noqa: F401
    import render  # noqa: F401
    import split  # noqa: F401
    import storyboard  # noqa: F401
    import tts  # noqa: F401


def test_split_whole_novel():
    from split import split

    cfg = load_cfg()
    data = split(Path(cfg["novel"]["path"]))
    assert data["book"]["chapters"] == 120
    assert data["book"]["chars"] > 150_000
    assert data["book"]["shots_estimate"] > 10_000


def test_storyboard_rule_path_uses_english_visuals():
    from storyboard import rule_based_shots

    scenes = [
        {"heading": "通知", "lines": ["同学群发通知的那天是周三。", "我在开会，手机在桌上亮了一下。我没看。"]},
    ]
    shots = rule_based_shots(scenes, ["陈屿", "林晚"])
    assert len(shots) == 2
    assert shots[0]["setting"].startswith("open plan office")
    assert "smartphone" in shots[1]["props"]
    assert shots[1]["action"] == "phone screen lighting up"
    # 中文原句应保留在 narration，不得被改写
    assert shots[0]["narration"] == "同学群发通知的那天是周三。"


def test_prompt_builder_locks_age_and_safety():
    from prompt_builder import PromptBuilder

    pb = PromptBuilder(load_cfg())
    assert set(pb.chars) == {"陈屿", "林晚"}

    pos, neg = pb.build({"shot_type": "close", "characters": [{"name": "陈屿"}], "mood": "平静"})
    assert pos.startswith("safe")
    assert "mature adult man" in pos
    assert "30 years old" in pos
    assert "nsfw" in neg
    assert "2girls" in neg


def test_comfy_workflow_structure_and_ipadapter_branch():
    from comfy_client import build_sdxl_workflow

    base = build_sdxl_workflow(ckpt_name="Illustrious-XL-v2.0.safetensors", positive="a", negative="b")
    assert base["1"]["class_type"] == "CheckpointLoaderSimple"
    assert base["5"]["class_type"] == "KSampler"
    assert base["5"]["inputs"]["model"] == ["1", 0]
    assert base["7"]["class_type"] == "SaveImage"

    ipa = build_sdxl_workflow(
        ckpt_name="Illustrious-XL-v2.0.safetensors",
        positive="a",
        negative="b",
        ipadapter_ref="ref_chenyu_head.png",
        ipadapter_weight=0.6,
    )
    assert ipa["5"]["inputs"]["model"] == ["21", 0]
    assert ipa["21"]["class_type"] == "IPAdapterAdvanced"
    assert ipa["30"]["inputs"]["image"] == "ref_chenyu_head.png"


@pytest.mark.parametrize("move", ["zoom_in", "zoom_out", "pan_left", "pan_right", "tilt_up", "tilt_down", "static"])
def test_zoompan_filter(move):
    from render import zoompan_filter

    f = zoompan_filter(move, 90, 30)
    assert "zoompan" in f
    assert "2160:3840" in f
    assert "1080x1920" in f


def test_comfyui_is_online():
    from comfy_client import ComfyClient

    if not ComfyClient().is_up():
        pytest.skip("ComfyUI 未启动")


def test_ffmpeg_available():
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg 不在 PATH")
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    assert out.returncode == 0
    subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)


def test_final_sample_video_spec():
    mp4 = ROOT / "output" / "ch001_sample.mp4"
    if not mp4.exists():
        pytest.skip("尚无成片")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-show_entries", "stream=width,height", "-of", "json", str(mp4)],
        capture_output=True, text=True, encoding="utf-8",
    )
    info = json.loads(out.stdout)
    stream = info["streams"][0]
    assert (stream["width"], stream["height"]) == (1080, 1920)
    assert 40 < float(info["format"]["duration"]) < 60
