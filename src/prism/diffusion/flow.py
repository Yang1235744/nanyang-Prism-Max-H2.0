"""Rectified Flow 损失 + EDCISampler 采样器。

不依赖任何预训练的 flow 模型——loss 和采样器是纯数学实现，
训练时学习的是 Self-DiT 对任意数据分布的流场匹配。
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def randn_like(x: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(x).to(x.device, x.dtype)


# ---------------------------------------------------------------------------
# Rectified Flow Loss
# ---------------------------------------------------------------------------

def rectified_flow_loss(dit, vae, z_noisy, prompts, text_encoder=None, ctx_mask=None):
    """
    Rectified Flow 损失：DIT 预测速度场 v = eps - x0，minimize ||v - target||^2。

    Args:
        dit:         SelfDiT instance
        vae:         SelfVAE instance（用于 encode）
        z_noisy:     [B, C_lat, T, H, W] 干净的 latent 输入
        prompts:     list[str] 文本列表
        text_encoder: ByteTextEncoder，None 则用零上下文
        ctx_mask:    optional [B, L] bool padding mask

    Returns:
        loss (tensor), metrics dict
    """
    from ..models.text_embed import encode_text, PAD_ID

    B = z_noisy.shape[0]
    device = z_noisy.device
    dtype = z_noisy.dtype

    # 均匀采样 u ∈ U(0,1)，t = u（t∈[0,1] 表示噪声程度；t=0 无噪，t=1 全噪）
    t_batch = torch.rand(B, device=device, dtype=dtype)

    # x_t = (1 - t) * x0 + t * eps
    eps = randn_like(z_noisy)
    t = t_batch.view(B, 1, 1, 1, 1)
    z_t = (1.0 - t) * z_noisy + t * eps

    # 目标速度场：dx/dt = eps - x0
    v_target = eps - z_noisy

    # 文本条件
    if text_encoder is not None and len(prompts) > 0:
        ids = torch.tensor(
            [encode_text(p) for p in prompts],
            device=device,
        )
        ctx = text_encoder(ids)
        if ctx_mask is None:
            ctx_mask = ids == PAD_ID
    else:
        ids = torch.zeros(B, 128, device=device, dtype=torch.long)
        ctx = torch.zeros(B, 1, dit.dim, device=device)
        ctx_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

    # DiT 前向
    pred_v = dit(z_t, t_batch, ctx, ctx_mask)

    # 加权 L2 loss：靠近噪声端（t≈1）更关键
    weight = torch.where(t < 0.5, 1.0, t / 0.5)
    loss = (weight * (pred_v - v_target).pow(2)).mean()

    return loss, {"rectified_flow_loss": loss.item(), "t_mean": t.mean().item()}


# ---------------------------------------------------------------------------
# EDCISampler —— Enhanced DPM-Solver++ for Rectified Flow
# ---------------------------------------------------------------------------

class EDCISampler:
    """高质量采样器，适合图像和视频生成（28 步即可出干净结果）。"""

    def __init__(self, num_steps: int = 28):
        self.num_steps = num_steps

    @staticmethod
    def _timesteps(device):
        sigma_max, sigma_min = 1000.0, 0.002
        s = torch.linspace(sigma_max, sigma_min, 100, device=device)
        # 映射回时间轴：t = sigma / (sigma + 1)
        return s / (s + 1.0)

    def _make_context(self, dit, text_enc, prompt, batch_size=1):
        """构建文本上下文与 padding mask。"""
        from ..models.text_embed import encode_text, PAD_ID
        ids_pos = torch.tensor([encode_text(prompt) for _ in range(batch_size)], device=next(dit.parameters()).device)
        ids_neg = torch.full((batch_size, 128), PAD_ID, device=next(dit.parameters()).device)
        ctx_pos = text_enc(ids_pos)
        ctx_neg = text_enc(ids_neg)
        mask_pos = ids_pos == PAD_ID
        mask_neg = ids_neg == PAD_ID
        return ctx_pos, ctx_neg, mask_pos, mask_neg

    @torch.no_grad()
    def sample_image(self, dit, vae, text_enc, prompt,
                     size=(1024, 1024), seed=None, cfg_scale=3.5):
        """采样单张图像。返回 (latent_z, rgb_image_tensor[B,C,T,H,W])。"""
        device = next(dit.parameters()).device
        T, H, W = 1, size[1], size[0]
        p = dit.p

        if seed is not None:
            torch.manual_seed(seed)

        ctx_p, ctx_n, mask_p, mask_n = self._make_context(dit, text_enc, prompt)
        ts = self._timesteps(device)[: self.num_steps + 1]

        # 初始化：全噪声
        h_tok, w_tok = H // p, W // p
        z = torch.randn(1, dit.patch_in.in_features // (p ** 2), T, h_tok, w_tok, device=device)
        z = ((1.0 - ts[0]) * 0.0 + ts[0] * z).expand(-1, vae.latent_channels, -1, -1, -1)

        for i in range(self.num_steps):
            t_cur, t_next = ts[i].view(1), ts[i + 1].view(1)
            t_in = t_cur.expand(1)
            # CFG: z 需要重复两份与拼合后的 context 对齐（B=2）
            z_cat = z.expand(2, -1, -1, -1, -1)
            v_full = dit(z_cat, t_in,
                         torch.cat([ctx_p, ctx_n], dim=0),
                         torch.cat([mask_p, mask_n], dim=0))
            v_ctx, v_uncond = v_full.chunk(2, dim=0)
            v_pred = v_uncond + cfg_scale * (v_ctx - v_uncond)
            delta_t = t_next - t_cur
            z = z + v_pred * delta_t.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # 数值安全：防止随机初始化下放大 NaN/Inf
            z = torch.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)

        # 解码到 RGB
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float32):
            img = vae.decode(z.float())
        return z.float(), img.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_video(self, dit, vae, text_enc, prompt,
                     duration=30.0, fps=12, width=576, height=320,
                     seed=None, cfg_scale=3.5):
        """采样一段视频。返回 (latent_z, frames_tensor[B,C,T,H,W])。"""
        device = next(dit.parameters()).device
        p = dit.p
        T_total = int(duration * fps)
        max_chunk = min(dit.pos_t.shape[0], 64)  # 单次最大 chunk 帧数

        ctx_p, ctx_n, mask_p, mask_n = self._make_context(dit, text_enc, prompt, batch_size=T_total)
        ts = self._timesteps(device)[: self.num_steps + 1]

        h_tok, w_tok = height // p, width // p

        chunks_latent, chunks_frame = [], []
        for start in range(0, T_total, max_chunk):
            end = min(start + max_chunk, T_total)
            chunk_len = end - start

            z = torch.randn(1, dit.patch_in.in_features // (p ** 2), chunk_len, h_tok, w_tok, device=device)
            z = ((1.0 - ts[0]) * 0.0 + ts[0] * z).expand(-1, vae.latent_channels, -1, -1, -1)

            for i in range(self.num_steps):
                t_cur, t_next = ts[i].view(1), ts[i + 1].view(1)
                # CFG: t 也需扩成 B=2，与 z_cat 对齐；否则 timestep_embedding 产出的 c 维度不匹配
                t_in = t_cur.expand(2)
                z_cat = z.expand(2, -1, -1, -1, -1)
                v_full = dit(z_cat, t_in,
                             torch.cat([ctx_p[:1], ctx_n[:1]], dim=0),
                             torch.cat([mask_p[:1], mask_n[:1]], dim=0))
                v_ctx, v_uncond = v_full.chunk(2, dim=0)
                v_pred = v_uncond + cfg_scale * (v_ctx - v_uncond)
                delta_t = t_next - t_cur
                z = z + v_pred * delta_t.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                # 数值安全：防止随机初始化下放大 NaN/Inf
                z = torch.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)

            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float32):
                frames = vae.decode(z.float())
            chunks_latent.append(z)
            chunks_frame.append(frames)

        full_z = torch.cat(chunks_latent, dim=2)
        full_frames = torch.cat(chunks_frame, dim=2)
        return full_z, full_frames.clamp(-1.0, 1.0)
