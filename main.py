"""CLI 入口：
    python main.py train --schedule linear --epochs 20     # 训练
    python main.py sample --ckpt checkpoints/unet_final.pt # 生成 8x8 网格
    python main.py sample --sample-steps 1000 100 50 --eta 0  # DDIM 少步采样对比
    python main.py check                                   # 自检
"""
import argparse

from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Awesome-Micro-Diffusion-Model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练 U-Net 去噪网络")
    train_parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--sample-every", type=int, default=0,
                              help="每多少个 epoch 采样监控一次（0=关闭）")
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--resume", nargs="?", const="./checkpoints/unet_latest.pt",
                              default=None, metavar="CKPT",
                              help="断点续训；不带路径时默认 checkpoints/unet_latest.pt")

    sample_parser = subparsers.add_parser("sample", help="从纯噪声采样生成样本网格")
    sample_parser.add_argument("--ckpt", default="./checkpoints/unet_final.pt")
    sample_parser.add_argument("--num-samples", type=int, default=64)
    sample_parser.add_argument("--out", default=None, help="输出图片路径")
    sample_parser.add_argument("--sample-steps", type=int, nargs="+", default=None, metavar="S",
                               help="反向采样步数，可传多个做质量-耗时对比（默认走满训练时的 T）")
    sample_parser.add_argument("--eta", type=float, default=1.0,
                               help="DDIM 随机性：0=确定性（少步时质量更好），1=DDPM 后验")

    subparsers.add_parser("check", help="自检：参数量/调度/形状/单batch过拟合")
    return parser.parse_args()


def build_config(args):
    """从默认 Config 出发，应用命令行覆盖。"""
    config = Config()
    if args.command == "train":
        config.schedule = args.schedule
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.lr is not None:
            config.lr = args.lr
        config.sample_every = args.sample_every
        if args.seed is not None:
            config.seed = args.seed
    return config


def main():
    args = parse_args()
    if args.command == "train":
        from engine.train import train
        train(build_config(args), resume_path=args.resume)
    elif args.command == "sample":
        from engine.sample import sample
        sample(build_config(args), args.ckpt, args.num_samples, args.out,
               args.sample_steps, args.eta)
    elif args.command == "check":
        from engine.check import run_check
        run_check(Config())


if __name__ == "__main__":
    main()
