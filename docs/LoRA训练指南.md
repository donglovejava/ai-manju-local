# 角色 LoRA 训练指南（8GB 实测）

> 硬件：RTX 4060 Laptop 8GB / 16GB RAM / Win11
> 实测：沈迟 LoRA，50 张素材，1200 步，**27 分 35 秒**，loss 收敛到 0.107
> 日期：2026-11-01

---

## 一、为什么不用 ComfyUI 内置的 TrainLoraNode

| 方案 | 每步耗时 | 1200 步 | 结论 |
|---|---|---|---|
| ComfyUI `TrainLoraNode` | **189 秒** | 60 小时 | ❌ 不可用 |
| kohya sd-scripts | **1.4 秒** | 27 分钟 | ✅ 采用 |

ComfyUI 的实现没有针对 8GB 做优化，`offloading=True` 时权重在 CPU/GPU 间反复搬运，速度崩到 1/100；`offloading=False` 直接 OOM（需 13.4GB）。

## 二、完整流程（四步）

### 1. 生成训练素材

```bash
python pipeline/lora_dataset.py --char 沈迟
```

- 12 种构图/姿态 × 5 个 seed = **60 张**
- 输出到 `assets/lora/<角色>/`
- 耗时约 25 分钟

**素材要点**：

| 要点 | 说明 |
|---|---|
| **发型必须露额** | 长发遮脸会让 LoRA 学不到五官。角色卡写 `forehead fully visible, no long hair`，负面词加 `hair over eyes, bangs covering eyes` |
| **构图要多样** | 全是立绘会导致 LoRA 只会画"三视图设定图"。下一轮应加入更多半身、环境互动、不同机位 |
| **排除明显崩坏** | 多人物拼贴、手部畸变、线条粗糙的，直接删 |

### 2. 筛选与准备数据集

```bash
# 排除已知问题组（示例：closeup 易长发遮脸，sitting_desk 线条粗糙）
cd assets/lora/沈迟
mkdir -p picked
for f in *.png; do case "$f" in *closeup*|*sitting_desk*) ;; *) cp "$f" picked/ ;; esac; done
```

kohya 数据集格式：

```
D:\manju\lora_dataset\1_shenchi\
├── 沈迟_front_s1101.png
├── 沈迟_front_s1101.txt      # caption
├── ...
```

`1_shenchi` 里的 `1_` 是重复次数（`num_repeats`），`shenchi` 是 class token。

caption 示例：

```
shenchi, a young Chinese man, 22 years old, short black hair,
dark navy wool coat, grey knit sweater, front view, facing camera directly
```

**关键**：触发词放最前面，然后是固定角色描述，最后是变化描述。

### 3. 训练

```bash
cd C:\project\manju

# 重要：先停掉 ComfyUI 释放显存
powershell.exe -NoProfile -Command 'Stop-Process -Id <comfy_pid> -Force'

export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/d/manju/ComfyUI_windows_portable/python_embeded/python.exe pipeline/run_sdxl_train.py \
  sdxl_train_network.py \
  --pretrained_model_name_or_path "D:\manju\ComfyUI_windows_portable\ComfyUI\models\checkpoints\Illustrious-XL-v2.0.safetensors" \
  --train_data_dir "D:\manju\lora_dataset" \
  --output_dir "D:\manju\lora_output" \
  --output_name shenchi_v1 \
  --network_module networks.lora \
  --network_dim 16 --network_alpha 8 \
  --resolution "512,896" \
  --train_batch_size 1 \
  --max_train_steps 1200 \
  --learning_rate 4e-4 \
  --optimizer_type Adafactor \
  --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False" \
  --lr_scheduler constant \
  --gradient_checkpointing \
  --mixed_precision fp16 \
  --save_precision fp16 \
  --cache_latents --cache_latents_to_disk \
  --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk \
  --network_train_unet_only \
  --max_data_loader_n_workers 1 \
  --sdpa \
  --save_every_n_steps 400 \
  --seed 42
```

**这组参数的每一行的作用**：

| 参数 | 作用 |
|---|---|
| `--network_train_unet_only` | 只训 UNet，不训 text encoder（省 3–4GB 显存） |
| `--cache_text_encoder_outputs_to_disk` | 把 caption 的 embedding 缓存到磁盘，训练时**不加载 TE** |
| `--cache_latents_to_disk` | 图像 latent 也缓存到磁盘 |
| `--gradient_checkpointing` | 用时间换显存 |
| `--optimizer_type Adafactor` | 优化器状态几乎不占显存（AdamW 要 2 倍模型大小） |
| `--sdpa` | PyTorch 2.x 的 attention 加速 |
| **不要 `--lowram`** | 名字骗人，它会把模型加载到 **VRAM**，反而更易 OOM |

### 4. 安装与使用

```bash
cp D:/manju/lora_output/shenchi_v1.safetensors \
   D:/manju/ComfyUI_windows_portable/ComfyUI/models/loras/

# 重启 ComfyUI，然后在出图时指定
python pipeline/run_episode.py --config config_wzn.yaml --chapter 1 \
  --max-shots 6 --lora shenchi_v1.safetensors
```

**推荐强度 0.6**（0.85 会导致幼态化和构图失控）。

## 三、踩过的七个坑

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | ComfyUI TrainLoraNode | 189 秒/步 | 换 kohya |
| 2 | sd-scripts main 分支 | `train_network.py` 缺失（仓库自身 bug） | 用 v0.12.3 tag |
| 3 | embedded Python | `PYTHONPATH` 不生效，`import train_network` 失败 | 写 wrapper 手动 `sys.path.insert` |
| 4 | ComfyUI 常驻 | 「6.63GB 空闲却分配不了 112MB」 | 训练前停掉 ComfyUI |
| 5 | `--lowram` | 名字省内存，实际把模型塞进 VRAM | 去掉 |
| 6 | SDXL 加载阶段 | UNet + 两个 CLIP 撑爆 8GB | `--cache_text_encoder_outputs_to_disk` |
| 7 | tokenizer 下载 | 卡在 huggingface.co | `HF_ENDPOINT=https://hf-mirror.com` |

## 四、效果与局限

**已解决**：跨镜头同一张脸。黑发、浓眉、瘦长脸、深蓝大衣、灰毛衣在多个场景保持一致。

**仍存在**：

| 问题 | 原因 | 改进 |
|---|---|---|
| 构图偏「立绘/站桩」 | 训练素材以立绘式为主，LoRA 把构图也学进去了 | 下轮素材加入更多半身、环境互动、不同机位 |
| 人物略偏年轻 | 部分素材偏少年感 | 加 `--keep_tokens`，或降强度到 0.5 |
| 道具不稳定（围巾、书包） | 素材中道具出现频率不固定 | 在 caption 里固定道具描述 |

## 五、下一轮训苏锦时的清单

1. `python pipeline/lora_dataset.py --char 苏锦`
2. 检查发型是否露额（女性角色尤其容易长发遮脸）
3. 排除 closeup / 粗糙构图
4. 复制到 `D:\manju\lora_dataset\1_sujin\`（**改目录名，避免和沈迟混在一起**）
5. caption 触发词用 `sujin`
6. 训练参数完全复用上表，只改 `--output_name sujin_v1`
7. 强度同样从 0.6 起测

## 六、文件位置速查

| 内容 | 路径 |
|---|---|
| 训练脚本（wrapper） | `pipeline/run_sdxl_train.py` |
| 素材生成 | `pipeline/lora_dataset.py` |
| 效果验证 | `pipeline/lora_test.py` |
| 原始素材 | `C:\project\manju\assets\lora\<角色>\` |
| 数据集 | `D:\manju\lora_dataset\1_<token>\` |
| LoRA 输出 | `D:\manju\lora_output\` |
| ComfyUI 模型目录 | `D:\manju\ComfyUI_windows_portable\ComfyUI\models\loras\` |
| sd-scripts | `D:\manju\sd-scripts-0.12.3\` |

---

## 七、双角色同框：已知限制与解法（实测）

### 实测结果

两个 LoRA 串联（`checkpoint → shenchi → sujin → KSampler`）测了 4 个双人场景：

| 场景 | 结果 |
|---|---|
| dual_classroom | ❌ 只画出一个人 |
| dual_corridor | ⚠️ 画出两个人，但**都是女性**（沈迟被苏锦 LoRA 覆盖） |
| dual_desk | ❌ 只画出一个人 |
| dual_snow | ❌ 只画出一个人 |

**结论：LoRA 串联无法可靠地把两个角色分配到两个人物上。** 后加载的 LoRA 会压过先加载的。

### 原因

两个 LoRA 都修改同一个 UNet 的同一批权重，特征互相干扰。模型没有「这是角色 A、那是角色 B」的空间概念。

### 解法（按实用性排序）

**1. 分镜规避（推荐，零成本）**

双人对话一律用**过肩镜头**（over-the-shoulder）：

```
前景：A 的肩膀/后脑勺虚化
背景：B 的侧脸或正面
```

每张图只需要一个清晰的脸。这是影视剧的标准手法，观众完全能接受。分镜阶段就该这样规划。

其他可用构图：
- 正反打（A 单独 → B 单独 → A 单独）
- 背影 + 正面
- 局部特写（A 的手递东西给 B 的手）
- 两人同框但都背对镜头 / 远景剪影

**2. Regional Prompter（分区提示词）**

把画面分左右两半，各自绑不同的提示词和 LoRA。需要装 `ComfyUI-Impact-Pack` 或 `sd-webui-regional-prompter` 的等价节点。效果比串联好，但配置复杂。

**3. Inpainting 合成**

先画 A，再用 inpainting 在指定区域补 B。质量控制最好，但每张双人图要两道工序。

**4. 训练关系 LoRA**

用「两人同框」的图训一个 LoRA，但需要先有大量同框图——鸡生蛋。

### 对分镜脚本的要求

`storyboard.py` 的 system prompt 应该增加规则：

> 同一镜头只允许一个角色出现正脸。需要两人对话时，拆成过肩镜头或正反打，不要让两个角色同框对视。

这条规则尚未加入，是下一步要做的改动。

---

## 八、动态强度实测：解决了一个问题，暴露了另一个

### 做法

`config.image.lora_strength_by_shot` 按景别给不同强度：

```yaml
extreme_close: 0.62   # 近景压低，防幼态
close: 0.65
medium: 0.75
wide: 0.85            # 全景拉高，锁形象
over_shoulder: 0.72
```

`run_episode.py` 的 `shot_lora_strength()` 读取该表，命令行 `--lora-strength` 可强制覆盖。

### 结果

| 项 | 结果 |
|---|---|
| 幼态化 | ✅ 解决（近景 0.62–0.65 不再画成小孩） |
| 全身/中景形象锁定 | ⚠️ 部分改善，但仍不稳定 |
| **跨镜头同一张脸** | ❌ **仍未达成** |

实测同一部片子、同为 `medium`（0.75 强度）的两帧：

- 第 6 秒：短发、圆脸、偏年轻
- 第 38 秒：中长发遮耳、瘦长脸、偏成熟

### 根因：训练素材内部矛盾

60 张素材里同时存在短发和中长发（抽检时 `profile` 是长直发、`upper_body` 是短齐肩）。LoRA 把两种发型都学了，出图时随机选一种。**参数压不住数据内部的矛盾。**

### 正确解法（按优先级）

1. **重筛素材，统一发型**（最根本）
   - 只保留与角色卡一致的短发图
   - 女性角色尤其要注意：齐肩 vs 中长必须二选一
   - 筛完重新训一版（27 分钟）

2. **在 caption 里强化发型描述**
   - 每张图都写 `short black hair, above the ears`
   - 让发型成为"必须满足的条件"而非"可能出现的特征"

3. **提高素材多样性但保持特征一致**
   - 角度、场景、光照要变
   - 发型、脸型、年龄不能变

### 一句话总结

> LoRA 学到的是你喂给它的东西。素材里有什么矛盾，出图就会有什么矛盾。调强度、加负面词都只是临时的止痛药。

---

## 九、两轮迭代后的方法论修正（重要）

### 苏锦：v1 → v2 明确提升

| 项 | v1（50 张全用） | v2（32 张筛选后） |
|---|---|---|
| 服装瑕疵 | 胸有色斑 | 干净 |
| 手部 | 藏在背后 | 十指交叠，结构正常 |
| 比例 | 一般 | 更瘦长自然 |
| 背景 | 纯灰墙 | 门框、走廊、窗 |
| 跨场景一致性 | 一般 | **脸/发型/服装完全一致** |

### 沈迟：v1 vs v2 各有优劣

| 项 | v1（50 张） | v2（48 张筛选后） |
|---|---|---|
| 脸的相似度 | —— | **同一张脸** |
| 比例 | 略歪 | 更准 |
| **背景** | **门口、地板、门框（有场景）** | **纯灰色** |
| 构图 | 倾斜、有透视 | 标准立绘站姿 |

### 关键教训：筛选标准错了

第二轮筛选时，我只做了「排除脏图」（双人、幼态、遮脸），结果**同时删掉了带场景的好图**：

- 删掉 `window_s1101`（双人，但背景有窗和街景）
- 删掉 `sitting_desk_s3303`（小孩，但有桌椅环境）

留下的 48 张里大部分是纯色背景的标准立绘，于是 v2 的 LoRA 学会了「沈迟 = 站在纯色背景前」。

**正确的筛选标准是三个维度同时满足**：

1. **单人** —— 排除双人/多人/拼贴
2. **成年** —— 排除幼态/圆脸/大头小身
3. **有场景** —— 背景里要有具体环境（房间、街道、窗、桌椅），不能是纯色

第三条是这次才意识到的。前两条决定「脸对不对」，第三条决定「会不会画成立绘」。

### 素材配比建议

| 类型 | 占比 | 作用 |
|---|---|---|
| 特写/半身（脸清晰） | 40% | 锁定五官 |
| 全身/中景（有场景） | 40% | 学会和环境互动 |
| 动作/特殊姿态 | 20% | 避免只会站桩 |

纯色背景的立绘最多占 20%，否则 LoRA 会产生强烈的「立绘倾向」。

### 一句话总结

> 你排除什么，LoRA 就少学什么。为了去掉双人图而连场景一起删掉，代价就是背景退化。筛选要「定向保留」，不能只「无差别排除」。
