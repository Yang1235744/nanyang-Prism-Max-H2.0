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

---

## 直接下载（不自己训练）

现成权重（SelfVAE + Self-DiT checkpoint）托管在自建 NAS，永久有效、无需注册：

```
https://share.fnnas.net/s/90de87a72789493597
```

下载后解压至 `~/.cache/nanyang-prism-h20/checkpoints/` 即可推理。

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
| v2.0.1 | 🔲 未来计划 | Flux / SDXL 双通道优化，批量推理加速 |
| v2.1 | 🔲 未来计划 | 口型同步 + 剧情对白自动生成 |
| v2.2 | 🔲 未来计划 | 批量生产流水线 |

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
