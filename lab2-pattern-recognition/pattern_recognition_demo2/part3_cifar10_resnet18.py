"""COMP3710 Lab 2 Part 3.2: CIFAR-10 自建 ResNet-18 与 mixed precision。"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True  # 输入尺寸固定时，cuDNN 会选择较快算法。


class BasicBlock(nn.Module):
    """ResNet 基本残差块：学习 F(x)，最后与 shortcut x 相加。"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 尺寸或通道改变时，identity 也要投影到相同形状才能相加。
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 残差连接让梯度更容易跨层传播，是 ResNet 能训练深网络的核心。
        return self.relu(out + identity)


class CIFARResNet18(nn.Module):
    """针对 32x32 CIFAR 图片调整的 ResNet-18，不调用预建模型。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64
        # ImageNet ResNet 的开头是 7x7/stride=2；CIFAR 很小，改用 3x3/stride=1。
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # [2, 2, 2, 2] 个 BasicBlock，每块 2 个卷积，即 ResNet-18。
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialise_weights()

    def _make_layer(self, out_channels: int, blocks: int, stride: int):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.extend(BasicBlock(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def _initialise_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)  # raw logits；CrossEntropyLoss 内部处理 softmax。


def create_loaders(data_dir: Path, batch_size: int, workers: int):
    # 随机裁剪和翻转只用于训练，增加数据变化以降低过拟合。
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.CIFAR10(data_dir, train=True, download=True,
                                 transform=train_transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=test_transform)
    common = dict(batch_size=batch_size, num_workers=workers,
                  pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = total_correct = total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # CUDA 上 float16 mixed precision 可加速并减少显存；CPU 时自动关闭。
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()  # 固定 BatchNorm 的统计量，并关闭训练专用行为。
    total_loss = total_correct = total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/part3_cifar10_resnet18.pt"))
    parser.add_argument("--demo", action="store_true",
                        help="加载 checkpoint，演示 inference 和一个训练 epoch")
    parser.add_argument("--smoke-test", action="store_true",
                        help="只跑一个 epoch 检查完整流程")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    train_loader, test_loader = create_loaders(args.data_dir, args.batch_size, args.workers)
    model = CIFARResNet18().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if args.demo:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Inference | test loss {test_loss:.4f} accuracy {test_acc:.3%}")
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        print(f"One demo epoch | loss {train_loss:.4f} accuracy {train_acc:.3%} "
              f"time {time.perf_counter() - start:.1f}s")
        return

    epochs = 1 if args.smoke_test else args.epochs
    # CosineAnnealing 将学习率平滑降到较小值，后期更容易稳定收敛。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_accuracy = 0.0
    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.checkpoint)

        print(f"Epoch {epoch:03d}/{epochs} | train {train_loss:.4f}/{train_acc:.2%} "
              f"| test {test_loss:.4f}/{test_acc:.2%} | "
              f"time {time.perf_counter() - epoch_start:.1f}s")

    print(f"Best test accuracy: {best_accuracy:.3%}")
    print(f"Total training time: {time.perf_counter() - training_start:.1f}s")
    print("Best checkpoint:", args.checkpoint)


if __name__ == "__main__":
    main()
