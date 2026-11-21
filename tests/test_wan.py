"""Wan 视频生成 workflow 与风格预设的结构回归测试（不实际出片）。"""
from wan_test import DEFAULT_NEG, STYLE_PRESETS, build_wan_workflow, style_prompt


def test_wan_workflow_node_types():
    wf = build_wan_workflow(positive="a man in the rain")
    assert wf["1"]["class_type"] == "UnetLoaderGGUF"
    assert wf["1"]["inputs"]["unet_name"].endswith(".gguf")
    assert wf["2"]["class_type"] == "CLIPLoader"
    assert wf["2"]["inputs"]["type"] == "wan"
    assert wf["3"]["class_type"] == "VAELoader"
    assert wf["6"]["class_type"] == "Wan22ImageToVideoLatent"
    assert wf["7"]["class_type"] == "KSampler"
    assert wf["8"]["class_type"] == "VAEDecode"
    assert wf["9"]["class_type"] == "SaveWEBM"


def test_wan_workflow_model_chain():
    wf = build_wan_workflow(positive="x")
    assert wf["7"]["inputs"]["model"] == ["1", 0]
    assert wf["7"]["inputs"]["positive"] == ["4", 0]
    assert wf["7"]["inputs"]["negative"] == ["5", 0]
    assert wf["7"]["inputs"]["latent_image"] == ["6", 0]
    assert wf["8"]["inputs"]["vae"] == ["3", 0]
    assert wf["9"]["inputs"]["images"] == ["8", 0]


def test_wan_workflow_dimensions_and_defaults():
    wf = build_wan_workflow(positive="x")
    lat = wf["6"]["inputs"]
    assert lat["width"] == 704 and lat["height"] == 1280
    assert lat["length"] == 49
    assert lat["batch_size"] == 1
    assert wf["5"]["inputs"]["text"] == DEFAULT_NEG
    assert wf["9"]["inputs"]["codec"] == "vp9"


def test_wan_workflow_honours_overrides():
    wf = build_wan_workflow(
        positive="p", negative="n", width=512, height=896,
        length=25, fps=16, steps=12, cfg=3.5, seed=7, prefix="demo",
    )
    assert wf["4"]["inputs"]["text"] == "p"
    assert wf["5"]["inputs"]["text"] == "n"
    assert wf["6"]["inputs"]["width"] == 512
    assert wf["6"]["inputs"]["height"] == 896
    assert wf["6"]["inputs"]["length"] == 25
    assert wf["7"]["inputs"]["steps"] == 12
    assert wf["7"]["inputs"]["cfg"] == 3.5
    assert wf["7"]["inputs"]["seed"] == 7
    assert wf["9"]["inputs"]["fps"] == 16.0
    assert wf["9"]["inputs"]["filename_prefix"] == "demo"


# ---------- 风格预设：锁住「要 3D 动漫，不要真人」这个需求 ----------

def test_style_presets_are_complete():
    assert set(STYLE_PRESETS) == {"donghua3d", "toon3d", "pixar"}
    for name, (pos, neg) in STYLE_PRESETS.items():
        assert "3D" in pos or "3d" in pos, f"{name} 正向缺少 3D 标识"
        assert "photorealistic" in neg, f"{name} 负向未排除写实"


def test_style_prompt_combines_subject_and_prefix():
    pos, neg = style_prompt("toon3d", "一个男人站在雨里")
    assert pos.startswith(STYLE_PRESETS["toon3d"][0])
    assert pos.endswith("一个男人站在雨里")
    assert DEFAULT_NEG in neg
    assert STYLE_PRESETS["toon3d"][1] in neg


def test_all_presets_push_away_from_live_action():
    """用户明确要「漫画 3D 人物，不是真人」，每个预设都必须压制真人观感。"""
    for name in STYLE_PRESETS:
        _, neg = style_prompt(name, "subject")
        for banned in ("photorealistic", "realistic", "live action", "real person"):
            assert banned in neg, f"{name} 负向缺少 {banned}"


def test_toon3d_preset_is_the_anime_one():
    pos, _ = style_prompt("toon3d", "x")
    assert "cel-shaded" in pos
    assert "anime" in pos
    assert "outlines" in pos
