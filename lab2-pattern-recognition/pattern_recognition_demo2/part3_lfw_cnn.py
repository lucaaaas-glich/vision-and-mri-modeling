"""COMP3710 Lab 2 Part 3.1: 使用两层 CNN 进行 LFW 人脸分类。"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    """固定随机数，方便重复实验和向老师解释结果。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_data(batch_size: int):
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # P2 使用 .data（已摊平）；CNN 必须使用 .images 来保留脸部的二维空间结构。
    X = lfw_people.images.astype(np.float32)
    y = lfw_people.target.astype(np.int64)
    target_names = lfw_people.target_names

    print(f"Pixel range: [{X.min():.3f}, {X.max():.3f}]")
    print("Original image array:", X.shape)

    # 使用与 P2 相同的 test_size 和 random_state，结果才有可比性。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Conv2d 接收 NCHW：[样本数, 通道数, 高, 宽]。灰度图的通道数为 1。
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("CNN train shape:", X_train.shape)
    print("CNN test shape:", X_test.shape)
    print("Classes:", list(target_names))
    return train_loader, test_loader, target_names, X.shape[1:]


class FaceCNN(nn.Module):
    """作业要求：两个 3x3、32-filter 的卷积层，再接 dense 分类层。"""

    def __init__(self, n_classes: int, image_shape: tuple[int, int]):
        super().__init__()
        self.features = nn.Sequential(
            # padding=1 且 stride=1，因此卷积后高和宽保持不变。
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 第二层输入通道必须等于上一层输出的 32 张 feature maps。
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # 用一次 dummy forward 自动计算 flatten 大小，避免手算尺寸写错。
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *image_shape)
            flattened_size = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
        # 这里不加 Softmax：CrossEntropyLoss 需要直接接收 raw logits。


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)  # 训练时启用 Dropout；测试时关闭 Dropout。
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if training:
            optimizer.zero_grad()  # 清除上一个 batch 留下的梯度。

        # 测试阶段不建立计算图，节省显存和计算时间。
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()    # 反向传播：计算每个参数的梯度。
                optimizer.step()   # Adam 根据梯度更新参数。

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions, labels = [], []
    for images, batch_labels in loader:
        logits = model(images.to(device))
        predictions.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(batch_labels.numpy())
    return np.asarray(labels), np.asarray(predictions)


def plot_history(history, output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["test_loss"], label="Test")
    axes[0].set(title="LFW CNN Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["test_accuracy"], label="Test")
    axes[1].set(title="LFW CNN Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--smoke-test", action="store_true", help="只训练 1 epoch，检查流程")
    args = parser.parse_args()

    set_seed()
    device = choose_device()
    print("Device:", device)
    train_loader, test_loader, target_names, image_shape = prepare_data(args.batch_size)

    model = FaceCNN(len(target_names), image_shape).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print(model)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    epochs = 1 if args.smoke_test else args.epochs
    history = {key: [] for key in ("train_loss", "test_loss", "train_accuracy", "test_accuracy")}
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_acc)
        history["test_accuracy"].append(test_acc)
        print(
            f"Epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} "
            f"acc {train_acc:.3%} | test loss {test_loss:.4f} acc {test_acc:.3%}"
        )

    y_true, y_pred = predict(model, test_loader, device)
    print("\nFinal classification report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), results_dir / "part3_lfw_cnn.pt")
    plot_history(history, results_dir / "part3_lfw_cnn_history.png")
    print("Saved model and learning curves to:", results_dir)


if __name__ == "__main__":
    main()
