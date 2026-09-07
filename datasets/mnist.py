"""数据集准备：MNIST 下载、归一化到 [-1,1]、DataLoader 构建。"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(config, train=True):
    # ToTensor() 把像素缩放到 [0,1]，再线性映射到 [-1,1]（扩散模型的标准输入范围）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    dataset = datasets.MNIST(root=config.data_dir, train=train, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=0,      # Windows/CPU 下最稳妥
        drop_last=train,
    )
