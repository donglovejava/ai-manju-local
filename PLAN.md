# 小说 → 漫剧 本地流水线方案

> 目标：把《暗恋那三年，她也在等》(约 20 万字符 / 15.9 万正文字 / 都市言情) 转成竖屏 9:16 动态漫画剧集
> 实测规模：120 章、13563 个镜头、成片总长约 11 小时；按 3 分钟/集约 220–230 集
> 形态：动态漫画（AI 出图 + 运镜 + 配音 + 字幕），非真·视频生成
> 模式：混合 —— 分镜拆解用 LLM API，出图 / 配音 / 合成全本地

## 一、硬件约束（实测）

| 项目 | 实测 | 约束含义 |
|---|---|---|
| GPU | RTX 4060 Laptop **8GB** VRAM | SDXL 原生可跑；视频模型需 FP8/GGUF 量化 |
| RAM | **16GB** | 硬瓶颈。出图与 TTS **必须错峰运行**，不能并发 |
| CPU | i9-13980HX 24C/32T | 可作为 offload 兜底 |
| 磁盘 | 954GB NVMe 单盘，C 剩 264G / D 剩 230G | 模型全部放 D 盘 |
| 系统 | Win11 家庭版 | ComfyUI portable 免安装，无需 WSL |

**必做前置**：把虚拟内存设到 D 盘 32–64GB（`系统属性 → 高级 → 性能 → 虚拟内存`）。16GB 物理内存在加载 SDXL + 浏览器 + 系统时会触发换页，没有大页面文件会直接 OOM 崩溃。

## 二、技术选型

| 环节 | 选型 | 理由 |
|---|---|---|
| 出图引擎 | ComfyUI portable（NVIDIA） | 节点化、可 API 调用、8GB 生态最成熟 |
| 底模 | **Illustrious-XL v2.0**（6.94GB） | 韩漫/日漫线条与上色最佳，SDXL 架构 8GB 可跑，社区 LoRA 生态最大 |
| 画风强化 | 韩漫/条漫 LoRA（Civitai 不通，走 HF / ModelScope） | 统一"干净线条 + 高级灰"的条漫质感 |
| 角色一致性 | 三件套：角色设定图 → IP-Adapter Plus → 角色 LoRA（可选训练） | 漫剧崩坏的第一杀手；先靠 IP-Adapter 零训练起步 |
| 放大 | RealESRGAN_x4plus_anime_6B + 4x-UltraSharp | 1024 → 1440/1920 不糊线条 |
| 运镜 | FFmpeg `zoompan` + 缓动曲线 | 推/拉/摇/移四种基础镜头，无需 3D |
| 特效 | FFmpeg overlay（雨、光斑、速度线 PNG 序列） | 成本极低，观感提升明显 |
| 配音 | **IndexTTS-2**（中文，情感参考音频） | 中文情感表现最好；备选 CosyVoice2-0.5B（更快更省） |
| 口型 | LatentSync 1.6（**可选**，后置） | 8GB 可跑；对话特写才用，全片用会拖慢 3 倍 |
| 字幕 | FFmpeg `subtitles` 烧录（ASS，手动调样式） | 条漫剧字幕是节奏的一部分，要逐句切 |
| 编排 | Python 3.11 + ffmpeg-python | 薄胶水层，不引入重框架 |

## 三、目录结构

```
C:\project\manju\              # 代码与剧本资产（工作区内）
├── PLAN.md
├── config.yaml                # 全局配置（路径、模型、API、画幅）
├── pipeline\
│   ├── split.py               # 小说 → 章节 → 场景切分
│   ├── storyboard.py          # 场景 → 分镜 JSON（调 LLM API）
│   ├── prompt_builder.py      # 分镜 → SDXL 提示词（角色卡 + 画风锁）
│   ├── comfy_client.py        # ComfyUI HTTP API 批量出图
│   ├── tts.py                 # 台词 → wav（IndexTTS-2）
│   ├── render.py              # 图 + 音 + 字幕 → mp4（FFmpeg）
│   └── run_episode.py         # 端到端编排
├── assets\
│   ├── chars\                 # 角色设定图 + 角色卡 yaml
│   ├── style\                 # 画风锁定词、负面词
│   └── sfx\                   # 音效、BGM、雨/光斑 overlay
├── work\{episode}\            # 中间产物（shots.json / images / audio / clips）
└── output\                    # 成片

D:\manju\                      # 大文件
├── ComfyUI\                   # portable 主程序
├── downloads\                 # 原始下载
├── models\                    # checkpoint / lora / vae / ipadapter（软链进 ComfyUI/models）
└── TTS\                       # IndexTTS-2 权重
```

## 四、流水线

```
小说章节 .md
   │  split.py        按「（场景名）」标记切场景（你的小说正文已用括号分场景，天然适配）
   ▼
scenes.json           {scene_id, 角色, 地点, 时间, 原文}
   │  storyboard.py    LLM API → 结构化分镜
   ▼
shots.json            每镜头: {id, 景别, 构图, 角色, 表情, 动作, 台词, 情绪, 时长, 转场}
   │  prompt_builder.py  角色卡 + 风格前缀 + 景别词 → 正向/负向提示词
   ▼
comfy_client.py       批出图 1024×1024（竖屏裁切）→ upscale 1440×2560
   ▼
tts.py                台词 → 带情绪的 wav（每句一条，便于对齐）
   ▼
render.py             zoompan 运镜 + 字幕 ASS + 音轨混合 + 转场 → 单集 mp4
   ▼
output\EP01.mp4
```

### 分镜节奏基准（竖屏漫剧）
- 每镜头 **2.5–4 秒**；对话镜头 2–3 秒，空镜/情绪镜头 4–5 秒
- 一集 3 分钟 ≈ **50–70 个镜头**
- 每 8–12 个镜头插一个"钩子"（特写 / 反转 / 强情绪）
- 前 5 秒必须出冲突或悬念（平台完播率决定生死）

## 五、8GB 显存的具体运行参数

| 场景 | 参数 |
|---|---|
| SDXL 出图 | `--normalvram`；1024×1024, 28 steps, DPM++ 2M Karras, CFG 5.5；约 **8–12 秒/张** |
| 放大 | 分块 upscale 到 1440×2560（约 15 秒/张） |
| IP-Adapter | 与底模同载，显存 +0.6GB，可接受 |
| TTS | 单独进程跑，先卸 ComfyUI 模型（`/free` 接口），避免与出图争显存 |
| 角色 LoRA 训练 | kohya_ss SDXL，`--lowvram --gradient_checkpointing`，1024 分辨率，20 张素材约 1.5–2 小时/角色 |

## 六、风险与对策

| 风险 | 对策 |
|---|---|
| **角色脸崩**（最大风险） | 实测结论：提示词+年龄锁只能保证"同类人"，**IP-Adapter Plus 在本组合下不可用**（性别反转 + 过曝偏色，见 README 第三节）。可行路径：① 训角色 LoRA（最可靠）② PuLID-SDXL ③ 接受风格一致、靠运镜与构图转移注意力 |
| 16GB 内存 OOM | 单进程串行跑；出图完 `/free` 释放；虚拟内存 48GB |
| 韩漫 LoRA 拿不到（Civitai 不通） | 先用 Illustrious 原生 + 提示词验证；同时从 HF/ModelScope 找 manhwa LoRA；实在没有就自训风格 LoRA（20–30 张） |
| 全本工期过长 | 实测每章 1300–1800 正文字、约 110–135 个镜头。一集 3 分钟 ≈ 60–70 镜 ≈ 1 章拆 2 集；全本约 220–230 集。**先做第 1 章前半段 1 分钟样片**，验收后再谈批产 |
| 中文排版/字幕审美 | 字幕用思源黑体 Bold，字号按 1080 宽 46–52px，描边 3px，底部安全区留 180px |

## 七、验收标准（样片）

1. 同一角色在 ≥10 个连续镜头里"能认出是同一个人"
2. 画风统一（线条粗细、上色、色调不跳）
3. 配音与字幕时间误差 < 0.2 秒
4. 运镜无抖动、无黑边、缩放不超采样
5. 1080×1920 / 30fps / H.264 / AAC，可直接上传
