"""Self-VAE 训练脚本入口。

用法示例：
    python scripts/train_vae.py \
        --data-root data/datasets/my_dataset \
        --epochs 5 \
        --batch-size 8 \
        --lr 1e-4 \
        --save-every 500 \
        --device cuda:0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import accelerate

from prism.models.self_vae import SelfVAE
from prism.training.vae_trainer import VAETwoStageTrainer


def make_dataloader(root: Path, batch_size: int, num_workers: int = 4):
    """从 ImageFolder 构建 dataloader（带标准数据增强）。"""
    transform = v2.Compose([
        v2.Resize(256),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    ds = ImageFolder(str(root), transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    pin_memory=True, drop_last=True)
    return dl


def main():
    parser = argparse.ArgumentParser(description="nanyang-Prism-Max-H2.0 VAE 训练")
    parser.add_argument("--data-root", type=Path, required=True, help="包含 subfolder images/ 的根目录")
    parser.add_argument("--ckpt-dir", type=Path, default=Path.home() / ".cache/nanyang-prism-h20/checkpoints/vae")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dim", type=int, default=128, help="Base channel dimension")
    parser.add_argument("--batch-size", "-b", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs-phase1", type=int, default=5, help="Phase 1 L1 epochs")
    parser.add_argument("--epochs-phase2", type=int, default=5, help="Phase 2 KL warmup epochs")
    parser.add_argument("--kl-start", type=float, default=1e-6)
    parser.add_argument("--kl-end", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = args.device
    ckpt_dir = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    vae = SelfVAE(in_channels=3, latent_channels=4, base_ch=args.dim, max_frames=64)
    trainer = VAETwoStageTrainer(vae, lr=args.lr, device=device)

    # Dataset
    print(f"[Info] 数据集: {args.data_root}")
    loader = make_dataloader(args.data_root, args.batch_size, args.num_workers)

    # Phase 1: L1 only
    print("[Step 1/2] Phase 1 - L1 重建训练...")
    trainer.train_phase_l1(loader, epochs=args.epochs_phase1)

    # Phase 2: KL warmup
    print("[Step 2/2] Phase 2 - KL 微调...")
    trainer.train_phase_kl(loader, epochs=args.epochs_phase2,
                           kl_start=args.kl_start, kl_end=args.kl_end,
                           warmup_steps=args.warmup_steps)

    # Save
    trainer.save(ckpt_dir / "vae_latest.pt")
    print("[Done] VAE 训练完成！下一步：训练 Self-DiT。")
    print(f"  命令: python scripts/train_dit.py --vae-checkpoint {ckpt_dir}/vae_latest.pt")


if __name__ == "__main__":
    main()
