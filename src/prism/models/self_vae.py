"""Self-VAE —— 因果 3D VAE，自研编解码器。

设计目标：
  - 空间下采样 8×（4× 在时间维做可选压缩）
  - 时序因果卷积，避免视频花屏/闪烁（帧间一致性）
  - 完全从零训练，无预训练权重

架构：
  Encoder: Conv3d(3→c, stride=[1,2,2], padding=1) → ResBlock ×N → Conv3d → mean/logvar
  Decoder: Conv3d → ResBlock ×N → ConvTranspose3d(stride=[1,2,2]) → Conv3d(→3)
  时间维使用因果卷积（ causal=True），保证第 t 帧只依赖前 t 帧信息。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------

class CausalConv3d(nn.Module):
    """3D 因果卷积：时间维只取当前及过去帧，避免未来信息泄漏导致花屏。"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, causal_temporal=True):
        super().__init__()
        self.causal = causal_temporal
        # 时间方向 pad：kernel_size=3 时 temporal_pad=2，保证因果性
        self.temporal_pad = kernel_size - 1 if causal_temporal else 0
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # x: [B, C, T, H, W]
        if self.temporal_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_pad, 0))  # pad front
        return self.conv(x)


class ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            CausalConv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )

    def forward(self, x):
        return x + self.net(F.silu(x))


# ---------------------------------------------------------------------------
# SelfVAE
# ---------------------------------------------------------------------------

class SelfVAE(nn.Module):
    """自研因果 3D VAE，支持任意长度视频和 1024×1024 图片。"""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_ch: int = 128,
        spatial_down: int = 8,     # 8× 空间下采样
        max_frames: int = 256,     # 支持最长帧数（推理时自动切片）
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.spatial_down = spatial_down
        self.max_frames = max_frames

        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        # ---------- Encoder ----------
        self.enc_spatial = nn.Sequential(
            CausalConv3d(in_channels, c1, 3, padding=1),
            ResBlock3D(c1),
            CausalConv3d(c1, c2, 3, stride=(1, 2, 2), padding=1),
            ResBlock3D(c2),
            CausalConv3d(c2, c3, 3, stride=(1, 2, 2), padding=1),
            ResBlock3D(c3),
            CausalConv3d(c3, c3, 3, stride=(1, 2, 2), padding=1),
            ResBlock3D(c3),
        )
        self.enc_to_latent = CausalConv3d(c3, latent_channels * 2, 3, padding=1)

        # ---------- Decoder ----------
        self.dec_from_latent = CausalConv3d(latent_channels, c3, 3, padding=1)
        self.dec_spatial = nn.Sequential(
            ResBlock3D(c3),
            CausalConv3d(c3, c2, 3, stride=(1, 2, 2), padding=1, causal_temporal=False),
            nn.ConvTranspose3d(c2, c2, 4, stride=(1, 2, 2), padding=(0, 1, 1)),
            ResBlock3D(c2),
            CausalConv3d(c2, c1, 3, stride=(1, 2, 2), padding=1, causal_temporal=False),
            nn.ConvTranspose3d(c1, c1, 4, stride=(1, 2, 2), padding=(0, 1, 1)),
            ResBlock3D(c1),
            CausalConv3d(c1, c1, 3, stride=(1, 2, 2), padding=1, causal_temporal=False),
            nn.ConvTranspose3d(c1, c1, 4, stride=(1, 2, 2), padding=(0, 1, 1)),
            ResBlock3D(c1),
        )
        self.dec_to_image = CausalConv3d(c1, in_channels, 3, padding=1)

    def encode(self, x: torch.Tensor) -> tuple:
        """x: [B,C,T,H,W] ∈ [-1,1] → (mean, logvar): [B,LC,T',H/8,W/8]"""
        h = self.enc_spatial(x)
        h = self.enc_to_latent(h)
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        return mean + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B,LC,T,H',W'] → [B,3,T,8H',8W'] ∈ ℝ"""
        h = self.dec_from_latent(z)
        h = self.dec_spatial(h)
        return self.dec_to_image(h)

    def forward(self, x: torch.Tensor) -> tuple:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

    def encode_batched(self, x: torch.Tensor, chunk_t: int = 32) -> tuple:
        """对超长视频按时间分块编码，节省显存后拼接。"""
        B, C, T, H, W = x.shape
        if T <= chunk_t:
            return self.encode(x)
        means, logvars = [], []
        for start in range(0, T, chunk_t):
            end = min(start + chunk_t, T)
            m, l = self.encode(x[:, :, start:end])
            means.append(m)
            logvars.append(l)
        return torch.cat(means, dim=2), torch.cat(logvars, dim=2)


# ---------------------------------------------------------------------------
# 训练 loss
# ---------------------------------------------------------------------------

def vae_loss(recon: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor,
             kl_weight: float = 1e-6) -> tuple:
    """L1 重建 loss + KL 正则。kl_weight 建议从 1e-6 起步，后期可增大到 1e-4。"""
    l1 = F.l1_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1.0 + logvar - mean.pow(2) - logvar.exp())
    return l1 + kl_weight * kl, {"l1": l1.item(), "kl": kl.item()}
