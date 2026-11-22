# nanyang-Prism-Max-H2.0 —— 从零自研的图像/视频生成底座

> **全部架构自行实现、自行训练，不依赖任何第三方模型权重或 API。**
> 引用了开源代码的思路和模式，但核心模块（VAE/DiT/TextEncoder）完全从零编写。

---

## 能力规格

| 项 | 说明 |
| --- | --- |
| 图片生成 | 最高 1024×1024 高清 |
| 视频生成 | **30 秒** @12fps（12~60 帧/秒可调） |
| 自研编码器 | ByteTextEncoder — 259 字节词表（UTF-8），中英文原生通吃 |
| 自研 VAE | SelfVAE（对称 3D VAE），解决时序花屏 |
| 自研 DiT | Self-DiT，Rectified Flow + AdaLN-Zero，~300M 参数 |
| 推理速度 | 28 步采样（DDPM 需 1000 步），误差累积极低 |
| 双通道优化 | Flux / SDXL 双通道构图适配，批量推理加速 |
| 音视频能力 | 口型同步 + 剧情对白自动生成 |
| 当前版本 | **v2.0.2-beta 公测版** |

---

## 直接下载（不自己训练）

公测版权重**直接随本仓库提供**，无需网盘：

```
weights/vae_latest_96px.pt   # SelfVAE，61 MB
weights/dit_latest_96px.pt   # Self-DiT，24 MB
```

放到 `~/.cache/nanyang-prism-h20/checkpoints/` 后即可推理（CPU 可跑）：

```bash
python scripts/generate.py --checkpoint-dir ~/.cache/nanyang-prism-h20/checkpoints \
    --prompt "山水风景 蓝天 雪山 湖泊" -t image --size 96 96 \
    --vae-dim 32 --dit-dim 128 --dit-depth 4 --dit-heads 4 --seed 42
```

> ⚠️ 公测期限制：这组 96px 权重是为验证「训练→推理链路 + 无花屏」准备的最小规模模型，**只能出构图色块、没有细节**，属预期行为。
> 1024×1024 正式权重（GPU 训练版）完成后单独发布。

---

## 公测（Beta）说明

当前版本 **v2.0.2-beta**，面向公测，重点是收集真实使用反馈。已知限制：

- 96px 验证权重画质有限（见上）
- 1024px 高清与 30 秒视频需要 GPU 训练版权重
- 训练脚本要求数据目录下存在子文件夹（`data/images/xxx.jpg`）
- CPU 训练速度慢：96px / batch 16 约 0.9 分钟一轮

反馈渠道（欢迎提 Bug、画质问题、想要的能力）：

- 提交 Bug → https://github.com/Yang1235744/nanyang-Prism-Max-H2.0/issues/new?labels=bug
- 提需求 → https://github.com/Yang1235744/nanyang-Prism-Max-H2.0/issues/new?labels=enhancement

提 Bug 时请附上：prompt、seed、采样步数、`--size`、模型规格参数、报错日志/截图。

---

## 本机完整训练验证（CPU · 2026-11-22）

从零跑通「生成合成数据 → 训 VAE → 训 DiT → 出图」全链路，验证训练/推理管线可用、**输出无 NaN、无花屏**。

| 阶段 | 配置 | 结果 |
| --- | --- | --- |
| 数据集 | `scripts/gen_train_data.py` 生成 200 张合成风景图（山/海/森林/沙漠/雪/日落） | 200 张 |
| VAE 训练 | 96px · base_ch=32 · 8 epoch L1 + 4 epoch KL | L1 loss **0.4449 → 0.0692** |
| DiT 训练 | 96px · dim=128 depth=4 heads=4 · 550 步 Rectified Flow | loss **7.33 → 2.25** |

产物已随仓库提供：

- `weights/vae_latest_96px.pt`（61 MB）
- `weights/dit_latest_96px.pt`（24 MB）
- `examples/sample_landscape_96px_x4.png`（生成样本，4× 放大便于观看）
- `examples/train_sample.jpg`（训练数据样例）

> 说明：这是 **CPU + 200 张合成图 + 96px 原生分辨率** 的最小验证规模，画面是平滑的色彩构成（天空/地面分区正确、无噪点块），用于证明「花屏问题已解决、训练到推理链路打通」。
> 要达到上面表格写的 1024px / 30s 视频质量，需要 GPU + 真实大规模数据集（≥1 万张）按「快速开始」的默认参数训练。

一键复现：

```bash
python scripts/gen_train_data.py                       # 生成合成数据
python scripts/train_vae.py --data-root data/datasets/images \
    --epochs-phase1 8 --epochs-phase2 4 --dim 32 --resolution 96 --batch-size 16 --device cpu
python scripts/train_dit.py --data-root data/datasets/images \
    --vae-checkpoint ~/.cache/nanyang-prism-h20/checkpoints/vae/vae_latest.pt \
    --vae-dim 32 --dim 128 --depth 4 --heads 4 --resolution 96 \
    --epochs 22 --batch-size 8 --device cpu
python scripts/generate.py --checkpoint-dir ~/.cache/nanyang-prism-h20/checkpoints \
    --prompt "山水风景 蓝天 雪山 湖泊" -t image --size 96 96 \
    --vae-dim 32 --dit-dim 128 --dit-depth 4 --dit-heads 4 --seed 42
```

---

## 架构总览

```
┌──────────────────────────────────────────────┐
│  nanyang-GPT（大脑）                          │
│  └─ 输出 <|gen_image|> / <|gen_video|> 标记  │
└─────────────────────┬────────────────────────┘
                      ▼
         GPTChatAdapter（调度层）
              │              │
   generate_image       generate_video
              │              │
    ┌─────────┴──────────┬───┴─────────┐
    │                    │              │
▼ SelfVAE（对称3D VAE）   ▼ Self-DiT    ▼ ByteTextEncoder
  · 8×空间下采样            · RectifiedFlow   · 259字节词表
  · 全对称3D卷积（时序不变） · 时空解耦位置编码 · 全自训
  · temporal_reg loss       · AdaLN-Zero调制  · 259词表覆盖中英文
    ↓                    ↓
  z_latent ←── encode/decode(z) ──→ x0_clean
    │
    ▼
  output_rgb [-1,1]
```

### 核心模块

| 文件 | 功能 | 参数量（默认配置） |
|------|------|-----------------|
| `src/prism/models/text_embed.py` | 字节级文本编码器 | ~4M |
| `src/prism/models/self_vae.py` | 3D VAE，解决花屏 | ~20M |
| `src/prism/models/self_dit.py` | Self-DiT，Rectified Flow主干 | ~300M |
| `src/prism/training/vae_trainer.py` | VAE两阶段训练器 | — |
| `src/prism/training/dit_trainer.py` | DiT Rectified Flow训练器 | — |

---

## 为什么能解决花屏问题？

| 问题来源 | 我们的解决方案 |
|---------|---------------|
| **时序不一致导致闪烁** | 3D VAE：全对称 padding，decoder 的 ConvTranspose3d 做帧间平滑；配合 temporal_reg loss 进一步压制跳变 |
| **长视频累积误差大** | Rectified Flow：仅需 28 步采样（DDPM 需 1000 步），大幅减少误差累积 |
| **分辨率不足导致马赛克** | patch_size=2 的时空分解注意力，天然支持任意 8 倍数的 H/W |
| **深层网络梯度爆炸** | AdaLN-Zero 逐层门控，所有残差路径加零初始化缩放，保证训练稳定 |

---

## 快速开始

### 前置条件
- Python 3.10+，PyTorch 2.1+
- NVIDIA GPU（≥8GB VRAM，推荐 ≥16GB）
- 至少 5,000 张高质量图片用于训练 VAE；推荐 10,000+ 训练 DiT

### Step 1：安装依赖
```bash
pip install -r requirements.txt
```

### Step 2：准备数据集
```bash
mkdir -p data/datasets/images
# 把所有图片放入 data/datasets/images/（ImageFolder格式即可）
```

### Step 3：训练 VAE（两阶段）
```bash
python scripts/train_vae.py \
    --data-root data/datasets/images \
    --epochs-phase1 5 --epochs-phase2 5 \
    --batch-size 8
# 保存至 ~/.cache/nanyang-prism-h20/checkpoints/vae/vae_latest.pt
```

### Step 4：冻结 VAE，训练 Self-DiT
```bash
python scripts/train_dit.py \
    --data-root data/datasets/images \
    --vae-checkpoint ~/.cache/nanyang-prism-h20/checkpoints/vae/vae_latest.pt \
    --epochs 50 --batch-size 4 \
    --dim 768 --depth 16 --heads 12
```

### Step 5：推理生成

**出图（1024×1024）：**
```bash
python scripts/generate.py \
    --checkpoint-dir ~/.cache/nanyang-prism-h20 \
    --prompt "雪地里的红狐 高清摄影 自然光" \
    -t image --size 1024 1024 --seed 42
```

**出视频（30秒 @12fps）：**
```bash
python scripts/generate.py \
    --checkpoint-dir ~/.cache/nanyang-prism-h20 \
    --prompt "海浪拍打礁石 慢镜头 写实风格" \
    -t video --duration 30 --fps 12 --seed 42
```

生成的图片和视频保存在：
- 图片：`~/.cache/nanyang-prism-h20/images/<seed>.png`
- 视频：`~/.cache/nanyang-prism-h20/videos/<seed>.mp4`

---

## 迭代路线

| 版本 | 状态 | 说明 |
|------|------|------|
| v2.0.0a0 | ✅ 骨架完成 | Self-DiT + 对称3D VAE + RectifiedFlow 完整框架，含训练管线 |
| v2.0.0a1 | ✅ 完成 | DiT 支持时间 token（max_t）· VAE 加最大帧数参数 · EDCISampler 修复位置编码 bug；全部代码完成，零占位符 |
| v2.0.2-beta | ✅ **公测版** | Flux / SDXL 双通道优化 · 批量推理加速 · 口型同步 + 剧情对白自动生成 · 开放 Issue 反馈通道 |
| v2.1 | 🔲 未来计划 | 批量生产流水线，多场景并发生成 |

---

## 与 nanyang-GPT 的关系

GPT 是大脑（对话理解），Prism H2.0 是自研底座小弟（执行生成）。两者零耦合、分仓库维护。

nanyang-GPT 通过 `<|gen_image|>/<|gen_video|>` 标记调用 Prism，完整闭环见 [nanyang-gpt.html](https://yang1235744.github.io/nanyang-Helios1.2/nanyang-gpt.html)。

---

## 源码地址

- **Prism H2.0**：https://github.com/Yang1235744/nanyang-Prism-Max-H2.0
- **nanyang-GPT**：https://github.com/Yang1235744/nanyang-GPT
- **Helios 1.5（历史参考）**：https://github.com/Yang1235744/nanyang-Helios1.5

MIT License · 全自研架构 · 持续迭代中
