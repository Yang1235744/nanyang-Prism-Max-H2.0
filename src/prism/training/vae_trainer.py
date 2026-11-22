"""Self-VAE 两阶段训练器。

Phase 1: L1 重建训练（快，~5k steps）——让编码器/解码器学会保真压缩/还原
Phase 2: KL 微调（慢，~5k steps）——让隐空间分布更接近 N(0,I)
"""
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from prism.models.self_vae import SelfVAE, vae_loss
from prism.models.text_embed import ByteTextEncoder


class VAETwoStageTrainer:
    """训练 SelfVAE：两阶段，第一阶段只做 L1，第二阶段加入 KL。"""

    def __init__(self, model: SelfVAE, lr: float = 1e-4, device: str = "cuda"):
        self.model = model.to(device)
        self.lr = lr
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.02)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000)
        self.epoch = 0
        self.step = 0

    def train_phase_l1(self, loader: DataLoader, epochs: int = 5, grad_clip: float = 1.0):
        """Phase 1：L1 重建训练，KL_weight=0。"""
        self.model.train()
        total_steps = 0
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0.0
            for step, batch in enumerate(loader):
                images = batch["image"].to(self.device, non_blocking=True)
                recon, mean, logvar = self.model(images)
                loss, logs = vae_loss(recon, images, mean, logvar, kl_weight=0.0)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                total_steps += 1
                self.step = total_steps
            print(f"  [Phase1 L1] Epoch {epoch+1}/{epochs}  "
                  f"step={total_steps}  loss={running_loss/(step+1):.4f}  "
                  f"time={(time.time()-epoch_start)/60:.1f}min")
        return self.step

    def train_phase_kl(self, loader: DataLoader, epochs: int = 5,
                       kl_start: float = 1e-6, kl_end: float = 1e-3, warmup_steps: int = 2000):
        """Phase 2：逐步增加 KL 权重。"""
        self.model.train()
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0.0
            for step, batch in enumerate(loader):
                images = batch["image"].to(self.device, non_blocking=True)
                recon, mean, logvar = self.model(images)
                # Linear warmup on KL weight
                progress = min(self.step / warmup_steps, 1.0)
                kl_w = kl_start + (kl_end - kl_start) * progress
                loss, logs = vae_loss(recon, images, mean, logvar, kl_weight=kl_w)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                self.step += 1
            print(f"  [Phase2 KL] Epoch {epoch+1}/{epochs}  "
                  f"step={self.step}  loss={running_loss/(step+1):.4f}  "
                  f"kl_w={kl_w:.2e}  time={(time.time()-epoch_start)/60:.1f}min")
        return self.step

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "version": "h2.0-vae",
        }, path)
        print(f"  ✓ VAE saved → {path}")

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.step = ckpt.get("step", 0)
        print(f"  ✓ VAE loaded from {path}  step={self.step}")
