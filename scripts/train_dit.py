"""Self-DiT Rectified Flow 训练入口。

用法示例：
    python scripts/train_dit.py \\
        --data-root data/datasets/images \\
        --vae-checkpoint .cache/nanyang-prism-h20/checkpoints/vae/vae_latest.pt \\
        --epochs 50 \\
        --batch-size 4 \\
        --lr 1e-4 \\
        --save-every 2000 \\
        --device cuda:0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from prism.models.self_vae import SelfVAE
from prism.models.self_dit import SelfDiT
from prism.models.text_embed import ByteTextEncoder
from prism.training.dit_trainer import DiTRectifiedFlowTrainer


class SimplePromptDataset(ImageFolder):
    """ImageFolder 包装，返回空 prompt（后续可接入真实 caption）。"""
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self._prompts = [""] * len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = self.loader(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image.unsqueeze(1), "prompt": self._prompts[idx]}  # [C,1,H,W]


def make_dataloader(root: Path, batch_size: int, num_workers: int = 4, resolution: int = 256) -> DataLoader:
    transform = v2.Compose([
        v2.Resize((resolution, resolution)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    ds = SimplePromptDataset(str(root), transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    pin_memory=False, drop_last=True)
    return dl


def main():
    parser = argparse.ArgumentParser(description="nanyang-Prism-Max-H2.0 DiT 训练")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True,
                        help="冻结的 VAE 权重路径（.pt 或 .safetensors）")
    parser.add_argument("--ckpt-dir", type=Path,
                        default=Path.home() / ".cache/nanyang-prism-h20/checkpoints/dit")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--max-t", type=int, default=64,
                        help="时间 token 数上限（决定最长可处理帧数）")
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="梯度累积步数（等效 batch = batch_size * grad_accum）")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--vae-dim", type=int, default=128, help="必须与训练 VAE 时的 --dim 一致")
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    device = args.device
    ckpt_dir = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] 数据集: {args.data_root} 分辨率={args.resolution}")
    loader = make_dataloader(args.data_root, args.batch_size, args.num_workers, args.resolution)
    loader.gradient_accum_steps = args.grad_accum  # 传给 trainer

    # 加载并冻结 VAE
    print(f"[Info] 加载冻结 VAE: {args.vae_checkpoint} (dim={args.vae_dim})")
    vae = SelfVAE(in_channels=3, latent_channels=4, base_ch=args.vae_dim)
    ckpt = torch.load(args.vae_checkpoint, map_location="cpu")
    # 兼容 .pt 和 .safetensors 格式
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    vae.load_state_dict(state_dict, strict=False)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    text_enc = ByteTextEncoder(dim=512, depth=4, heads=4).to(device).eval()

    # 构建并训练 DiT
    dit = SelfDiT(in_channels=4, dim=args.dim, depth=args.depth,
                  heads=args.heads, patch_size=2, max_t=args.max_t)
    trainer = DiTRectifiedFlowTrainer(dit, vae, text_enc, lr=args.lr,
                                       grad_clip=1.0,
                                       amp_dtype=torch.bfloat16 if str(device).startswith("cuda") else None,
                                       device=device)

    if args.resume:
        trainer.load(args.resume)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        avg = trainer.train_one_epoch(loader, epoch)
        if avg < best_loss:
            best_loss = avg
            trainer.save(ckpt_dir / "dit_best.pt", best=True)
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            trainer.save(ckpt_dir / f"dit_ep{epoch + 1:04d}.pt")

    trainer.save(ckpt_dir / "dit_latest.pt")
    print(f"\n[Done] DiT 训练完成！保存到 {ckpt_dir}")
    print(f"下一步: python scripts/generate.py --checkpoint-dir {ckpt_dir.parent} --prompt '...\\' -t video --duration 30")


if __name__ == "__main__":
    main()
