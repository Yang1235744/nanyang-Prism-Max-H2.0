"""Self-VAE —— 自研 3D VAE，支持任意长度视频和高清图片。

设计目标：
  - 空间下采样 8x，潜层通道数 4
  - 时序维度全程严格不变（使用 AvgPool3d(kernel=(1,2,2)) + Upsample((1,2,2))）
  - Encoder/Decoder 对称，输入 T 帧 -> 输出恰好 T 帧
  - 两阶段训练：Phase1 L1重建 -> Phase2 KL正则化（时序平滑）
  - 完全从零训练，无预训练权重
  - 修复 H1.5 花屏问题

关键决策：
  - 不用 3D 卷积做空间下采样（会导致 T 维被意外改变或 T=1 时崩溃）
  - 改用 AvgPool3d(kernel=(1,2,2), stride=(1,2,2)) 做纯空间下采样
  - Decoder 用 nn.Upsample(scale_factor=(1,2,2)) 精确恢复，T 维度零变化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------

class ResBlock3d(nn.Module):
    """3D 残差块：两个 3x3x3 卷积 + GroupNorm + SiLU，仅做通道内变换，不改尺寸。"""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(F.silu(x))


# ---------------------------------------------------------------------------
# SelfVAE
# ---------------------------------------------------------------------------

class SelfVAE(nn.Module):
    """自研 3D VAE。Encoder/Decoder 严格对称，时序维度全程不变。

    架构说明：
      - Encoder 使用 AvgPool3d(kernel=(1,2,2),stride=(1,2,2)) 做空间下采样，
        T 方向 kernel=1/stride=1 => T 永远不变
      - Decoder 使用 nn.Upsample(scale_factor=(1,2,2)) 做空间上采样，
        T 方向 scale=1 => T 永远不变
      - 每个 stage 先 Conv3d(stride=1) 切换通道，再 Pool/Upsample 改空间尺寸
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_ch: int = 128,
        spatial_down: int = 8,
        max_frames: int = 256,
    ):
        super().__init__()
        assert spatial_down == 8, "本实现固定支持 8x 空间下采样"
        self.latent_channels = latent_channels
        self.spatial_down   = spatial_down
        self.max_frames     = max_frames

        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        # ---------- Encoder: 3 stages, each halves H/W, never touches T ----------
        self.enc_stage1 = nn.Sequential(
            nn.Conv3d(in_channels, c1, 3, stride=1, padding=1),
            ResBlock3d(c1),
        )
        self.enc_pool1 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc_stage2 = nn.Sequential(
            nn.Conv3d(c1, c2, 3, stride=1, padding=1),
            ResBlock3d(c2),
        )
        self.enc_pool2 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc_stage3 = nn.Sequential(
            nn.Conv3d(c2, c3, 3, stride=1, padding=1),
            ResBlock3d(c3),
        )
        self.enc_pool3 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc_to_latent = nn.Conv3d(c3, latent_channels * 2, 3, padding=1)

        # ---------- Decoder: mirror of encoder ----------
        self.dec_from_latent = nn.Conv3d(latent_channels, c3, 3, padding=1)

        self.dec_up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec_stage3 = nn.Sequential(
            ResBlock3d(c3),
            nn.Conv3d(c3, c3, 3, stride=1, padding=1),
            ResBlock3d(c3),
        )

        self.dec_up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec_stage2 = nn.Sequential(
            ResBlock3d(c3),
            nn.Conv3d(c3, c2, 3, stride=1, padding=1),
            ResBlock3d(c2),
        )

        self.dec_up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec_stage1 = nn.Sequential(
            ResBlock3d(c2),
            nn.Conv3d(c2, c1, 3, stride=1, padding=1),
            ResBlock3d(c1),
        )

        self.dec_to_image = nn.Conv3d(c1, in_channels, 3, padding=1)

    def encode(self, x: torch.Tensor) -> tuple:
        """x: [B,C,T,H,W] in [-1,1] -> (mean, logvar): [B,LC,T,H/8,W/8]"""
        h = self.enc_stage1(x)
        h = self.enc_pool1(h)        # H/W /2
        h = self.enc_stage2(h)
        h = self.enc_pool2(h)        # H/W /4
        h = self.enc_stage3(h)
        h = self.enc_pool3(h)        # H/W /8, T unchanged throughout
        h = self.enc_to_latent(h)
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        return mean + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B,LC,T,H',W'] -> [B,3,T,8H',8W']"""
        h = self.dec_from_latent(z)
        h = self.dec_stage3(h)
        h = self.dec_up3(h)          # H'/W' *2
        h = self.dec_stage2(h)
        h = self.dec_up2(h)          # H'/W' *4
        h = self.dec_stage1(h)
        h = self.dec_up1(h)          # H'/W' *8
        return self.dec_to_image(h)

    def forward(self, x: torch.Tensor) -> tuple:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

    def encode_batched(self, x: torch.Tensor, chunk_t: int = 32) -> tuple:
        """对超长视频按时间分块编码，节省显存后拼接（T 不变）。"""
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
             kl_weight: float = 1e-4, temporal_reg: bool = False) -> tuple:
    """L1 重建 loss + KL 正则 + 可选时序平滑项。

    kl_weight 默认 1e-4（H1.5 为 1e-6，太小导致潜空间过散、解码后出现随机伪影）。
    temporal_reg=True 时在损失中加入相邻帧 L2 差值惩罚，进一步抑制花屏。
    """
    l1 = F.l1_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1.0 + logvar - mean.pow(2) - logvar.exp())
    temp_smooth = torch.tensor(0.0, dtype=recon.dtype, device=recon.device)
    if temporal_reg and recon.shape[2] > 1:
        diff = recon[:, :, 1:, :, :] - recon[:, :, :-1, :, :]
        temp_smooth = diff.pow(2).mean()
    return l1 + kl_weight * kl + 1e-3 * temp_smooth, {
        "l1": l1.item(),
        "kl": kl.item(),
        "temp_smooth": temp_smooth.item() if temporal_reg else 0.0,
    }
