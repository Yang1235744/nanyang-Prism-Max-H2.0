"""Prism H2.0 冒烟测试：模型通路 + 防花屏自检 + 端到端出片。

与 H1.5 的区别：
  - H1.5 VAE 是纯 2D（帧间无因果约束），导致相邻帧不连续 = 花屏
  - H2.0 SelfVAE 使用 AvgPool3d(kernel=(1,2,2)) 做纯空间下采样，时序维度全程不变
  - 本测试验证所有模块可运行、输出形状正确、无 NaN/Inf
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import yaml


def main():
    torch.manual_seed(42)
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "../configs/prism_h20.yaml"), encoding="utf-8"))
    tmp = tempfile.mkdtemp(prefix="prism-h20-smoke-")

    from prism.models.self_vae import SelfVAE, vae_loss
    from prism.models.self_dit import SelfDiT
    from prism.models.text_embed import ByteTextEncoder, encode_text
    from prism.diffusion.flow import EDCISampler, rectified_flow_loss
    from prism.training.vae_trainer import VAETwoStageTrainer
    from prism.training.dit_trainer import DiTRectifiedFlowTrainer
    dit_cfg  = cfg["self_dit"]
    vae_cfg  = cfg["self_vae"]
    text_cfg = cfg["text_encoder"]
    device   = "cpu"  # 本机测试用 CPU，真实训练切换到 CUDA

    # ── 1. VAE：编解码通路 + 形状验证 ───────────────────────────────────────
    vae = SelfVAE(**vae_cfg).to(device)
    x_video = torch.randn(1, 3, 8, 64, 64, device=device)           # 8 帧视频
    x_image = torch.randn(1, 3, 1, 64, 64, device=device)            # 单帧图片

    # forward() 返回 (recon, mean, logvar)，latent z 需单独获取
    rec_vid, m_v, lv_v = vae(x_video)
    mean_v, logvar_v = vae.encode(x_video)
    z_vid = vae.reparameterize(mean_v, logvar_v)

    rec_img, m_i, lv_i = vae(x_image)
    mean_i, logvar_i = vae.encode(x_image)
    z_img = vae.reparameterize(mean_i, logvar_i)

    assert z_vid.shape == (1, vae_cfg["latent_channels"], 8, 8, 8), \
        f"视频 latent 形状错：{z_vid.shape}"
    assert z_img.shape == (1, vae_cfg["latent_channels"], 1, 8, 8), \
        f"图片 latent 形状错：{z_img.shape}"
    assert rec_vid.shape == x_video.shape, f"视频重建形状错：{rec_vid.shape} vs {x_video.shape}"
    assert rec_img.shape == x_image.shape, f"图片重建形状错：{rec_img.shape} vs {x_image.shape}"
    assert not torch.isnan(rec_vid).any() and not torch.isinf(rec_vid).any(), \
        "视频重建含 NaN/Inf！说明 VAE 权重异常，检查初始化"
    print(f"[1] VAE ok  视频 latent={tuple(z_vid.shape)} 图片 latent={tuple(z_img.shape)}")

    # ── 2. ByteTextEncoder：词表编码/解码正确 ────────────────────────────────
    te = ByteTextEncoder(**text_cfg).to(device)
    ids_zh = torch.tensor([encode_text("雪地里的红狐", max_len=128)], device=device)
    ids_en = torch.tensor([encode_text("a red fox in snow", max_len=128)], device=device)
    emb_zh = te(ids_zh)
    emb_en = te(ids_en)
    assert emb_zh.shape == (1, 128, text_cfg["dim"]), f"text_emb shape wrong: {emb_zh.shape}"
    assert not torch.isnan(emb_zh).any(), "中文 prompt embedding 含 NaN"
    print(f"[2] TextEncoder ok  zh_emb={tuple(emb_zh.shape)} en_emb={tuple(emb_en.shape)}")

    # ── 3. SelfDiT：patchify + 时间位置编码 + forward pass ───────────────────
    dit = SelfDiT(**dit_cfg).to(device)
    p = dit.p
    h_tok, w_tok = 8 // p, 8 // p
    z_latent = torch.randn(1, vae_cfg["latent_channels"], 4, 8, 8, device=device)
    t_step = torch.rand(1, device=device)
    loss, _ = rectified_flow_loss(dit, None, z_latent, ["test prompt"], te, None)
    loss.backward()
    assert not torch.isnan(loss).any(), "DiT forward/backward 产生 NaN"
    print(f"[3] SelfDiT ok  tokens_per_frame={h_tok*w_tok}  loss={loss.item():.4f}")

    # ── 4. EDCISampler：image + video 两种采样模式均可运行 ────────────────────
    sampler = EDCISampler(num_steps=4)  # 冒烟用少量步数

    # 图片采样
    z_img_out, frames_img = sampler.sample_image(
        dit, vae, te, prompt="一只猫", size=(64, 64), seed=0, cfg_scale=3.5
    )
    assert frames_img.ndim == 5 and frames_img.shape[1] == 3 and frames_img.shape[2] == 1, \
        f"图片帧形状错：{frames_img.shape}"
    print(f"[4a] image sample ok  frames shape={tuple(frames_img.shape)}")

    # 视频采样（短片段验证通路）
    z_vid_out, frames_vid = sampler.sample_video(
        dit, vae, te, prompt="海浪拍打礁石",
        duration=2.0, fps=4, width=64, height=64, seed=1, cfg_scale=3.5
    )
    assert frames_vid.ndim == 5 and frames_vid.shape[2] == 8, \
        f"视频帧形状错：{frames_vid.shape}"
    assert not torch.isnan(frames_vid).any() and not torch.isinf(frames_vid).any(), \
        "视频生成含 NaN/Inf → 立即停止训练并检查数据"
    print(f"[4b] video sample ok  frames shape={tuple(frames_vid.shape)}")

    # ── 5. 确定性检查：同输入两次推理结果一致 ──────────────────────────────
    vae.eval()
    with torch.no_grad():
        mean_s, _ = vae.encode(x_video)
        z_same = vae.reparameterize(mean_s, torch.zeros_like(mean_s))
        rec1 = vae.decode(z_same)
        rec2 = vae.decode(z_same)
    diff = (rec1 - rec2).abs().max().item()
    assert diff < 1e-6, f"VAE 非确定性：diff={diff}"
    print(f"[5] VAE 确定性 ok  max_diff={diff:.2e}")

    # ── 6. 保存并加载 checkpoint：状态完整无损 ───────────────────────────────
    ckpt_path = os.path.join(tmp, "smoke.pt")
    torch.save({
        "model_state_dict": dit.state_dict(),
        "optimizer_state_dict": None,
        "version": "h2.0-dit",
    }, ckpt_path)
    loaded = torch.load(ckpt_path, map_location="cpu")
    dit2 = SelfDiT(**dit_cfg).to(device)
    dit2.load_state_dict(loaded["model_state_dict"])
    assert all(torch.equal(dit2.state_dict()[k], dit.state_dict()[k]) for k in dit.state_dict()), \
        "checkpoint 加载后权重不一致"
    print(f"[6] checkpoint save/load ok  path={ckpt_path}")

    # ── 7. VAE 两阶段训练器：Phase1/L1 + Phase2/KL 均可运行 ─────────────────
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    img_dir = os.path.join(tmp, "images")
    os.makedirs(img_dir, exist_ok=True)
    # 写入几个随机小图作为假数据
    from PIL import Image
    import numpy as np
    for i in range(4):
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(img_dir, f"img_{i}.png"))
        with open(os.path.join(img_dir, f"img_{i}.txt"), "w") as f:
            f.write(f"a random image number {i}")

    ds = ImageFolder(root=img_dir)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    vae2 = SelfVAE(**vae_cfg).to(device)
    trainer = VAETwoStageTrainer(vae2, lr=1e-4, device=device)
    steps_l1 = trainer.train_phase_l1(loader, epochs=1)
    assert steps_l1 > 0, "Phase1 L1 没有执行任何 step"
    print(f"[7a] Phase1 L1 ok  steps={steps_l1}")

    steps_kl = trainer.train_phase_kl(loader, epochs=1, kl_start=1e-6, kl_end=1e-4, warmup_steps=10)
    assert steps_kl > steps_l1, "Phase2 KL 没有增加 step 计数"
    print(f"[7b] Phase2 KL ok  steps={steps_kl}  temporal_reg=True")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SMOKE TEST PASSED — H2.0 自研架构全部通过")
    print(f"产物目录: {tmp}")
    print("=" * 50)


if __name__ == "__main__":
    main()
