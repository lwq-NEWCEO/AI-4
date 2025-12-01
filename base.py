import os
# 解决OpenMP运行时库重复加载的问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# --- 1. 定义设置随机种子的函数 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 2. 定义imshow辅助函数 ---
def imshow(img):
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
    std = torch.tensor((0.2023, 0.1994, 0.2010)).view(3, 1, 1)
    img = img * std + mean  # 反标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 转换为 (H, W, C) 格式


# --- 3. 定义基线模型 ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 4. 定义训练函数 ---
def train_model(model, train_loader, val_loader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    return history


# --- 主程序入口 ---
if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理与加载（已下载数据集，故设置download=False）
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_base = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,  # 已存在数据集，无需重复下载
        transform=transform_base
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,  # 已存在数据集，无需重复下载
        transform=transform_base
    )

    train_size = 45000
    val_size = 5000
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 数据探索：显示样本图像
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    fig = plt.figure(figsize=(10, 10))
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        imshow(images[i])
        ax.set_title(classes[labels[i]], color="blue", fontsize=10)

    plt.suptitle("CIFAR-10 Dataset Sample Images", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig('cifar10_sample_images.png', dpi=300)
    plt.show()

    print("\n图像样本已保存为 'cifar10_sample_images.png'")

    # 类别分布检查
    print(f"\nCIFAR-10 训练集总样本数: {len(full_train_dataset)}")
    print(f"CIFAR-10 测试集总样本数: {len(test_dataset)}")
    print(f"CIFAR-10 类别数量: {len(classes)}")
    print(f"每个类别的样本数量: {len(full_train_dataset) // len(classes)} 张")

    # 训练基线模型
    model = SimpleCNN().to(device)
    print("\n开始训练 Baseline Model...")
    history_baseline = train_model(model, train_loader, val_loader, epochs=50)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_baseline['train_loss'], label='Train Loss')
    plt.plot(history_baseline['val_loss'], label='Val Loss')
    plt.title('Loss History (Baseline)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history_baseline['train_acc'], label='Train Accuracy')
    plt.plot(history_baseline['val_acc'], label='Val Accuracy')
    plt.title('Accuracy History (Baseline)')
    plt.legend()
    plt.savefig('baseline_training_history.png', dpi=300)
    plt.show()