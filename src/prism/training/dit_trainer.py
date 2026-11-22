"""Self-DiT Rectified Flow 训练器。

冻结 VAE 后，训练 Self-DiT 学习将纯噪声还原为有意义数据的 Rectified Flow。
支持图像批量、视频序列训练，支持梯度累积与 EMA。
"""
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prism.diffusion.flow import rectified_flow_loss
from prism.models.text_embed import ByteTextEncoder


class DiTRecurrentEMA:
    """Running exponential moving average of model weights。"""

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters()}

    def update(self, model):
        dn = 1.0 - self.decay
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(1 - dn).add_(param.detach(), alpha=dn)

    def apply(self, model):
        """临时将 shadow 复制到 model 上（用于验证/采样）。"""
        current = {name: p.data.clone() for name, p in model.named_parameters()}
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])
        return current

    def restore(self, current):
        for name, p in model.named_parameters():
            p.data.copy_(current[name])


class DiTRectifiedFlowTrainer:
    """Self-DiT Rectified Flow 训练主循环。"""

    def __init__(
        self,
        dit_model,
        vae,
        text_encoder,
        lr: float = 1e-4,
        ema_decay: float = 0.9999,
        grad_clip: float = 1.0,
        amp_dtype=None,
        device: str = "cuda",
    ):
        self.dit = dit_model.to(device).train()
        self.vae = vae.to(device).eval()
        self.text_encoder = text_encoder.to(device).eval()
        self.grad_clip = grad_clip
        self.amp_dtype = amp_dtype
        self.device = device

        # 优化器：只对 DiT 参数做动量，AdamW + Cosine LR
        self.optimizer = nn.optim.AdamW(
            dit_model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.02
        )
        self.scheduler = nn.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100_000)
        self.ema = DiTRecurrentEMA(dit_model, decay=ema_decay)
        self.step = 0
        self.best_loss = float("inf")

    def _text_to_ids(self, prompts, batch_size):
        """将一批 prompt 编码成固定长度的 ID 张量。"""
        from prism.models.text_embed import encode_text, PAD_ID
        max_len = 128
        ids_list = []
        masks = []
        for p in prompts:
            ids = encode_text(p, max_len=max_len)
            ids_list.append(ids)
            masks.append([False if i != PAD_ID else True for i in ids])
        ids_tensor = torch.tensor(ids_list, device=self.device)
        masks_tensor = torch.tensor(masks, dtype=torch.bool, device=self.device)
        return ids_tensor, masks_tensor

    @torch.no_grad()
    def evaluate(self, loader, epoch, wandb_log=None):
        """在训练集子集上评估，打印指标并记录 EMA 采样。"""
        self.dit.eval()
        losses = []
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 4:
                break
            images = batch["image"].to(self.device, non_blocking=True)
            prompts = batch["prompt"]
            ids_t, mask_t = self._text_to_ids(prompts, len(prompts))
            ctx = torch.cuda.amp.autocast(dtype=self.amp_dtype) if self.amp_dtype else nullcontext()
            with ctx:
                z_noisy = self.vae.encode(images)
                loss, logs = rectified_flow_loss(self.dit, self.vae, z_noisy, prompts, self.text_encoder, mask_t)
            losses.append(logs["rectified_flow_loss"])

        avg = sum(losses) / max(len(losses), 1)
        print(f"    [Eval] epoch={epoch} step={self.step}  loss={avg:.4f}")
        if wandb_log is not None:
            wandb_log.log({"val/loss": avg, "val/step": self.step}, step=self.step)
        self.dit.train()
        return avg

    def train_one_epoch(self, loader, epoch, val_loader=None, wandb_log=None):
        start = time.time()
        self.dit.train()
        ep_losses, ep_count = [], 0
        gc = torch.cuda.amp.autocast(dtype=self.amp_dtype) if self.amp_dtype else nullcontext()

        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(self.device, non_blocking=True)
            prompts = batch["prompt"]
            ids_t, mask_t = self._text_to_ids(prompts, len(prompts))

            with gc:
                z_noisy = self.vae.encode(images)
                loss, logs = rectified_flow_loss(self.dit, self.vae, z_noisy, prompts, self.text_encoder, mask_t)
                loss_scaled = loss / max(1, loader.gradient_accum_steps) if hasattr(loader, 'gradient_accum_steps') else loss

            self.optimizer.zero_grad(set_to_none=True)
            loss_scaled.backward()
            nn.utils.clip_grad_norm_(self.dit.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.ema.update(self.dit)
            self.step += 1

            ep_losses.append(logs["rectified_flow_loss"])
            ep_count += 1

        avg = sum(ep_losses) / max(ep_count, 1)
        elapsed = time.time() - start
        eta = elapsed / ep_count * (len(loader) - ep_count)
        if wandb_log is not None:
            wandb_log.log({"train/loss": avg, "train/step": self.step}, step=self.step)
        print(f"  epoch={epoch:3d}  step={self.step:6d}  "
              f"loss={avg:.4f}  t/s={ep_count/(elapsed or 1):.1f}  eta={eta/60:.1f}min")

        # Periodic evaluation
        if val_loader is not None and epoch % 5 == 0:
            self.evaluate(val_loader, epoch, wandb_log)

        return avg

    def save(self, path, best=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": (self.ema.shadow if best else self.dit.state_dict()),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.step,
            "ema_decay": self.ema.decay,
            "version": "h2.0-dit",
        }
        torch.save(state, path)
        tag = "best" if best else "latest"
        print(f"  ✓ DiT {tag} → {path}  step={self.step}")
        return path

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        try:
            self.dit.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError:
            # EMA 快照格式不同，从 shadow 加载
            self.ema.shadow = {k: v.cuda() for k, v in ckpt["model_state_dict"].items()}
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.step = ckpt.get("step", 0)
        print(f"  ✓ DiT loaded from {path}  step={self.step}")
