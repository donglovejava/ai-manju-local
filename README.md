# 8GB 笔记本上跑出 98 集 AI 漫剧

> 用一台 RTX 4060 Laptop（**8GB 显存**）+ 16GB 内存的笔记本，把一部 8 万字小说做成 **98 集竖屏动态漫画**。
> 全程本地出图、本地配音、本地合成，API 只用来做分镜拆解。
> 总成本约 **30 元**。

---

## 这个项目解决了什么问题

市面上做 AI 漫剧的方案大多是：

- 买商业平台额度（单集几十到几百元）
- 租云端 A100（一天几百元）
- 或者干脆做不了，因为本地显卡不够

这个项目证明：**8GB 显存足够做 2D 动态漫剧**，只要你选对技术路线。

### 核心判断：漫剧的「动」不需要视频生成模型

| 运动类型 | 实现方式 | 8GB 可行性 |
|---|---|---|
| 镜头推拉摇移 | FFmpeg `zoompan` 几何变换 | ✅ 秒级、零成本 |
| 画面元素动（眨眼、发丝） | 逐帧生成 | ❌ |
| 角色动作（走、转身） | 视频模型 | ❌ |

头部平台用后两种，所以「栩栩如生」。
但**第一种已经足够讲故事**——观众追漫剧看的是剧情和情绪。

### 实测数据：视频模型在 8GB 上不可行

用 Wan 2.2 TI2V-5B（Q4_K_M GGUF）实测：

| 配置 | 5 秒片段耗时 |
|---|---|
| 704×1280 / 49 帧 / 30 步 | **1770 秒（29.5 分钟）** |
| 512×896 / 25 帧 / 20 步 | 约 90 秒 |

一集 70 镜 × 29.5 分钟 = 34 小时/集。98 集需要 11 年。

**所以本项目走「静态图 + 几何运镜」路线。**

---

## 成品规格

| 项目 | 数值 |
|---|---|
| 成片 | 98 集（连续无缺口） |
| 单集时长 | 140–170 秒 / 约 70 镜 |
| 分辨率 | 1080 × 1920（竖屏 9:16） |
| 编码 | H.264 + AAC / 30fps |
| 素材量 | 6,862 张图 / 6,862 段配音 |
| 小说原文 | 117 章 / 80,506 汉字 |
| 纯生产耗时 | 约 40 小时（单集约 30 分钟） |

---

## 流水线

```
小说 md
  ↓ split.py          按场景标记切分
  ↓ storyboard.py     调用 LLM 生成分镜 JSON（每集 70 镜）
  ↓ prompt_builder.py 组装 danbooru 标签 + 角色卡 + 风格
  ↓ comfy_client.py   ComfyUI + Illustrious-XL + 角色 LoRA 出图
  ↓ tts.py            Piper 本地 TTS
  ↓ render.py         zoompan 运镜 + 环境音 + ASS 字幕
  ↓ driver.py         断点续跑批量生产
  output/epNNN.mp4
```

### 技术栈

| 环节 | 选型 | 理由 |
|---|---|---|
| 出图 | ComfyUI + **Illustrious-XL v2.0** | 对 danbooru 标签响应最好，条漫质感强 |
| 角色一致性 | **kohya_ss 训练角色 LoRA** | IP-Adapter 实测失败（性别翻转 + 过曝） |
| 配音 | **Piper**（ONNX） | 不需要 torch，63MB 模型，完全离线 |
| 分镜 | DeepSeek API | 唯一使用云服务的一步 |
| 运镜/合成 | FFmpeg | 零成本 |

---

## 关键经验

### 1. LoRA 强度存在「性别翻转阈值」

男主沈迟的实测：

| 强度 | 结果 |
|---|---|
| 0.55 | 女性 ❌ |
| **0.65** | **男性 ✅ 单人 ✅** |
| 0.72 | 女性 ❌ |

**0.65 是精确最优值**，高了低了都会翻性别。
换角色、换底模、换景别都必须重新标定。

### 2. IP-Adapter Plus 在 SDXL 上不可用

实测结果：男角色变女性、画面严重过曝、绿紫色偏。
根因是它迁移整体图像特征（含性别），不是纯人脸。

### 3. 双人同框必须用 over_shoulder

不给模型「两个正脸同框」的机会。
分镜阶段强制规则：多人镜头一律过肩构图，实测 100% 合规。

### 4. 参数必须显式传

`run_episode.py --max-shots` 默认 20，批量生产必须传 70。
本项目收尾时曾因忘传参数，把一集 70 镜覆盖成 20 镜。

---

## 快速开始

### 环境要求

- 显卡：**8GB 显存**以上（NVIDIA）
- 内存：**16GB** 以上（SDXL + 浏览器会吃满，建议虚拟内存 32GB+）
- 系统：Windows / Linux
- 磁盘：模型约 15GB + 中间产物约 30GB/集

### 安装

```bash
# 1. 安装 ComfyUI（推荐 portable 版）
# 2. 下载底模 Illustrious-XL v2.0 → ComfyUI/models/checkpoints/
# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. Piper 中文模型
pip install "piper-tts[zh]"
# 模型下载到 D:/TTS/piper/ 或任意目录

# 5. 配置
cp config.example.yaml config.yaml
# 修改 paths.comfyui 指向你的 ComfyUI 目录
```

### 运行

```bash
# 启动 ComfyUI（另开窗口）
python ComfyUI/main.py --port 8188

# 切分小说
python pipeline/split.py --config config.yaml

# 生成分镜（需要 .env 里的 DEEPSEEK_API_KEY）
python pipeline/storyboard.py --config config.yaml --chapter 1 --llm

# 出片
python pipeline/run_episode.py --config config.yaml \
  --shots-file work/eps/ep001_shots.json --episode 1 --max-shots 70

# 批量生产（断点续跑）
python pipeline/driver.py --config config.yaml
```

---

## 已知局限

1. **角色一致性依赖精确参数**——LoRA 强度窗口很窄
2. **全身构图偶发比例失调**——腿过长，负面词只能压低概率
3. **画面是静态图 + 几何运镜**——没有逐帧动画和口型同步
4. **没有 BGM**——只有 FFmpeg 合成的环境音（雨/风/雪/室内/街道）
5. **与头部平台有差距**——红果《全球死考》是 AI 写实 CG 视频，用了视频模型 + 高显存，不是同一技术路线

---

## 目录结构

```
.
├── pipeline/           # 21 个 Python 脚本
│   ├── split.py        # 小说切分
│   ├── storyboard.py   # 分镜生成
│   ├── prompt_builder.py
│   ├── comfy_client.py # ComfyUI API
│   ├── tts.py          # 5 种 TTS 引擎
│   ├── render.py       # 运镜/字幕/混音
│   ├── run_episode.py  # 单集执行
│   ├── batch.py        # 多集编排
│   ├── driver.py       # 断点续跑
│   └── lora_*.py       # LoRA 训练与对比
├── docs/               # 详细文档
├── tests/              # 41 项回归测试
├── assets/chars_wzn/   # 角色卡示例
├── config.example.yaml
└── README.md
```

---

## 文档

| 文档 | 内容 |
|---|---|
| [原理与调研](docs/原理与调研.md) | 为什么不用即梦、完整流程、GitHub 生态 |
| [红果漫剧拆解](docs/红果漫剧拆解.md) | 头部作品逐帧分析 |
| [Wan 本地视频实测](docs/Wan本地视频实测.md) | 三种 3D 风格实测数据 |
| [LoRA 训练指南](docs/LoRA训练指南.md) | 8GB 训练角色 LoRA 全流程 |
| [制作记录](docs/我去找你-制作记录.md) | 本书配置与踩坑 |

---

## License

（待定）

---

## 致谢

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Illustrious-XL](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0)
- [kohya_ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- [Piper](https://github.com/rhasspy/piper)
