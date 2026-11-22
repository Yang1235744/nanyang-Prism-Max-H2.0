"""推理入口：从本地 checkpoint 使用自研 Self-DiT + Self-VAE 生成图像/视频。

用法：
    python scripts/generate.py --prompt "雪地里的红狐" -t image --size 1024 1024
    python scripts/generate.py --prompt "海浪拍打礁石" -t video --duration 30 --fps 12

权重从 --checkpoint-dir 目录加载（默认 ~/.cache/nanyang-prism-h20/models/）。
模型架构完全自研，无任何第三方模型依赖。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from prism.models.self_vae import SelfVAE
from prism.models.self_dit import SelfDiT
from prism.models.text_embed import ByteTextEncoder
from prism.diffusion.flow import EDCISampler


def build_models(checkpoint_dir: Path, device: str):
    """加载 SelfVAE + SelfDiT + ByteTextEncoder。"""
    cache = checkpoint_dir / "models"
    cache.mkdir(parents=True, exist_ok=True)

    # VAE
    vae = SelfVAE(in_channels=3, latent_channels=4, base_ch=128)
    vae_path = cache / "vae_latest.pt"
    if not vae_path.exists():
        raise FileNotFoundError(f"未找到 VAE 权重: {vae_path}\n请先运行: python scripts/train_vae.py ...")
    vae.load_state_dict(torch.load(vae_path, map_location="cpu"), strict=False)

    # DiT
    dit = SelfDiT(in_channels=4, dim=768, depth=16, heads=12, patch_size=2, max_t=64)
    dit_path = cache / "dit_latest.pt"
    if not dit_path.exists():
        raise FileNotFoundError(f"未找到 DiT 权重: {dit_path}\n请先运行: python scripts/train_dit.py ...")
    dit.load_state_dict(torch.load(dit_path, map_location="cpu"), strict=False)

    # Text Encoder
    text_enc = ByteTextEncoder(dim=512, depth=4, heads=4)
    te_path = cache / "text_encoder.pt"
    if te_path.exists():
        text_enc.load_state_dict(torch.load(te_path, map_location="cpu"), strict=False)

    vae = vae.to(device).eval()
    dit = dit.to(device).eval()
    text_enc = text_enc.to(device).eval()
    print(f"  OK  VAE={vae_path.name}  DiT={dit_path.name}")
    return vae, dit, text_enc


@torch.no_grad()
def generate_image(args):
    device = args.device
    vae, dit, text_enc = build_models(args.checkpoint_dir, device)
    sampler = EDCISampler(num_steps=args.steps)

    _, img = sampler.sample_image(
        dit, vae, text_enc,
        prompt=args.prompt,
        size=args.size,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
    )
    out = (img + 1.0) * 0.5  # [-1,1] -> [0,1]
    out_path = args.out or (Path.home() / ".cache/nanyang-prism-h20/images" / f"{args.seed or 0}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    arr = (out[0].mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
           .permute(1, 2, 0).numpy())
    Image.fromarray(arr).save(out_path)
    print(f"[Prism H2.0] 图片已保存: {out_path.resolve()}")


@torch.no_grad()
def generate_video(args):
    device = args.device
    vae, dit, text_enc = build_models(args.checkpoint_dir, device)
    sampler = EDCISampler(num_steps=args.steps)

    _, frames = sampler.sample_video(
        dit, vae, text_enc,
        prompt=args.prompt,
        duration=args.duration,
        fps=args.fps,
        width=args.size[0],
        height=args.size[1],
        seed=args.seed,
        cfg_scale=args.cfg_scale,
    )
    out_path = args.out or (Path.home() / ".cache/nanyang-prism-h20/videos" / f"{args.seed or 0}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    videos_list = ((frames[0].mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
                     .permute(1, 2, 3, 0)).numpy()).tolist()
    imageio.mimsave(str(out_path), videos_list, fps=args.fps, codec="libx264")
    print(f"[Prism H2.0] 视频（{args.duration}秒）已保存: {out_path.resolve()}")


def main():
    p = argparse.ArgumentParser(description="nanyang-Prism-Max-H2.0 自研生成器")
    p.add_argument("--checkpoint-dir", default=str(Path.home() / ".cache/nanyang-prism-h20"))
    p.add_argument("--prompt", required=True)
    p.add_argument("-t", "--type", dest="output_type", default="image", choices=["image", "video"])
    p.add_argument("--size", type=int, nargs=2, default=[1024, 1024], metavar=("W", "H"))
    p.add_argument("--steps", type=int, default=28, help="采样步数（越高越精细，默认28）")
    p.add_argument("--cfg-scale", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--duration", type=float, default=30.0, help="视频时长（秒），默认30")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.output_type == "image":
        generate_image(args)
    else:
        generate_video(args)


if __name__ == "__main__":
    main()
