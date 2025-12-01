import os

# 解决OpenMP运行时库重复加载的问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # 导入学习率调度器


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


# --- 3. 定义基础模型（保留原版本，可选使用） ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 变为 16x16
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # pool 变为 8x8
        # Block 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # pool 变为 4x4

        # 全连接层
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)  # 输出10类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 4. 定义改进后的模型（添加BatchNorm2d） ---
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # Block 1: Conv → BN → ReLU → Pool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 添加Batch Normalization
        self.pool = nn.MaxPool2d(2, 2)

        # Block 2: Conv → BN → ReLU → Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 添加Batch Normalization

        # Block 3: Conv → BN → ReLU → Pool
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # 添加Batch Normalization

        # 全连接层
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 5. 定义改进的训练函数（添加学习率调度+L2正则） ---
def train_model(model, train_loader, val_loader, epochs=50):  # epochs提升至50
    criterion = nn.CrossEntropyLoss()
    # 优化器：增加学习率至0.01，添加L2正则化（weight_decay=5e-4）
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # 学习率调度器：余弦退火（T_max=epochs，学习率随epoch周期性衰减）
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

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

        # 计算训练集指标
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 验证阶段（禁用梯度计算）
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

        # 计算验证集指标
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        # 更新学习率（每个epoch结束后调整）
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率

        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # 打印日志（包含当前学习率）
        print(f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    return history


# --- 主程序入口 ---
if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 数据预处理与加载（训练集增强，验证/测试集仅标准化） ---
    # 训练集变换（添加数据增强）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪32x32，边缘填充4像素
        transforms.RandomHorizontalFlip(),  # 50%概率水平翻转
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 随机颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    # 验证集/测试集变换（无增强，仅基础预处理）
    transform_test_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    # 加载原始训练集（仅用于划分索引，不实际存储数据）
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=None
    )
    # 划分训练集/验证集索引（45k训练，5k验证）
    train_indices, val_indices = random_split(
        range(len(full_train_dataset)), [45000, 5000]
    )

    # 训练集：使用增强变换+训练集索引
    train_dataset = Subset(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train),
        train_indices.indices
    )
    # 验证集：使用无增强变换+验证集索引
    val_dataset = Subset(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test_val),
        val_indices.indices
    )
    # 测试集：无增强变换
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test_val
    )

    # DataLoader配置
    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # --- 数据探索性分析（EDA） ---
    # 显示训练集样本（注意：增强后的样本会随机变化，每次显示可能不同）
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    fig = plt.figure(figsize=(10, 10))
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        imshow(images[i])
        ax.set_title(classes[labels[i]], color="blue", fontsize=10)

    plt.suptitle("CIFAR-10 Augmented Training Samples", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig('cifar10_augmented_samples.png', dpi=300)
    plt.show()

    print("\n增强后的数据样本已保存为 'cifar10_augmented_samples.png'")

    # 类别分布检查
    print(f"\nCIFAR-10 训练集总样本数: {len(train_dataset)}")
    print(f"CIFAR-10 验证集总样本数: {len(val_dataset)}")
    print(f"CIFAR-10 测试集总样本数: {len(test_dataset)}")
    print(f"CIFAR-10 类别数量: {len(classes)}")
    print(f"每个类别的训练集样本数量: {len(train_dataset) // len(classes)} 张")

    # --- 训练改进后的模型 ---
    model_refined = ImprovedCNN().to(device)
    print("\n开始训练 Refined Model（数据增强+BatchNorm+学习率调度）...")
    history_refined = train_model(model_refined, train_loader, val_loader, epochs=50)

    # 绘制训练曲线（Loss + Accuracy + Learning Rate）
    plt.figure(figsize=(18, 5))

    # 1. 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history_refined['train_loss'], label='Train Loss', color='#1f77b4')
    plt.plot(history_refined['val_loss'], label='Val Loss', color='#ff7f0e')
    plt.title('Loss History (Refined CNN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    # 2. 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(history_refined['train_acc'], label='Train Accuracy', color='#1f77b4')
    plt.plot(history_refined['val_acc'], label='Val Accuracy', color='#ff7f0e')
    plt.title('Accuracy History (Refined CNN)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(alpha=0.3)

    # 3. 学习率调度曲线
    plt.subplot(1, 3, 3)
    plt.plot(history_refined['lr'], label='Learning Rate', color='#2ca02c')
    plt.title('Learning Rate Schedule (Cosine Annealing)')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('refined_cnn_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- （可选）测试集最终评估 ---
    print("\n开始在测试集上评估模型...")
    model_refined.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_refined(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f"测试集最终准确率: {test_acc:.2f}%")