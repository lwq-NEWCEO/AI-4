import os
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

# 导入新版权重枚举（如果不兼容旧版torchvision，代码会自动处理）
try:
    from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
except ImportError:
    from torchvision.models import efficientnet_b2

    EfficientNet_B2_Weights = None

# --- 配置环境 ---
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --- 1. 核心工具函数 ---
def set_seed(seed=42):
    """固定全链路随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=15, min_delta=1e-4, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter % 5 == 0:
                print(f"⚠️  早停计数器: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"🚫 触发早停！最佳验证损失: {self.best_loss:.4f}")


# --- 2. 模型定义 (EfficientNet-B2) ---

class EfficientNetTransfer(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetTransfer, self).__init__()

        print("🔄 正在加载 EfficientNet-B2 预训练权重...")
        # 兼容新旧版本 torchvision
        if EfficientNet_B2_Weights is not None:
            weights = EfficientNet_B2_Weights.DEFAULT
            self.base_model = efficientnet_b2(weights=weights)
        else:
            self.base_model = efficientnet_b2(pretrained=True)

        # EfficientNet 的分类头通常叫 'classifier'
        # 结构通常是: Dropout -> Linear
        # 我们需要获取原本 Linear 的输入特征数
        # B2 的 classifier[1] 是 Linear 层
        original_fc = self.base_model.classifier[1]
        in_features = original_fc.in_features

        # 重构分类头
        # EfficientNet 官方通常使用 Dropout=0.3
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

    def freeze_layers(self, freeze=True):
        """冻结/解冻 特征提取层"""
        # 冻结所有层
        for param in self.base_model.parameters():
            param.requires_grad = not freeze

        # 始终解冻分类头 (classifier 部分)
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True


# --- 3. 训练流程函数 ---

def train_stage_model(model, train_loader, val_loader, epochs, optimizer, scheduler, early_stopping, stage_name):
    # 使用标签平滑，这在 EfficientNet 论文中被强烈推荐
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0.0

    print(f"\n===== 开始 {stage_name}（共 {epochs} Epochs）=====")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Metrics
        avg_train_loss = train_loss / train_total
        avg_train_acc = 100 * train_correct / train_total
        avg_val_loss = val_loss / val_total
        avg_val_acc = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc

        print(f"Ep {epoch + 1}/{epochs} | LR: {current_lr:.2e} | "
              f"Tr Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        history['lr'].append(current_lr)

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            break

        scheduler.step()

    return history


def evaluate_test_set(model, test_loader, classes, device):
    model.eval()
    test_correct = 0
    test_total = 0
    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))

    print("\n🔍 正在评估测试集...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    acc = 100 * test_correct / test_total
    print(f"\n📊 测试集最终准确率: {acc:.2f}%")
    print("-" * 30)
    for i in range(len(classes)):
        if class_total[i] > 0:
            print(f"  {classes[i]:<10}: {100 * class_correct[i] / class_total[i]:.2f}%")
    print("-" * 30)
    return acc


def plot_history(hist1, hist2, filename='effnet_b2_history.png'):
    loss = hist1['train_loss'] + hist2['train_loss']
    val_loss = hist1['val_loss'] + hist2['val_loss']
    acc = hist1['train_acc'] + hist2['train_acc']
    val_acc = hist1['val_acc'] + hist2['val_acc']
    lr = hist1['lr'] + hist2['lr']

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.axvline(x=len(hist1['train_loss']) - 1, color='r', linestyle='--', alpha=0.5, label='Stage 1 End')
    plt.title('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.axvline(x=len(hist1['train_acc']) - 1, color='r', linestyle='--', alpha=0.5, label='Stage 1 End')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(lr, label='LR', color='green')
    plt.axvline(x=len(hist1['lr']) - 1, color='r', linestyle='--', alpha=0.5, label='Stage 1 End')
    plt.title('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\n📈 曲线图已保存至 {filename}")


# --- 主程序 ---
if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # 1. 数据准备
    # [关键优化] EfficientNet 需要稍大的分辨率才能发挥性能
    # 将 32x32 Resize 到 64x64，能显著提升准确率
    RESIZE_SIZE = 64

    transform_train = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),  # 放大图片
        transforms.RandomCrop(RESIZE_SIZE, padding=8),  # 适应更大的图
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.2)
    ])

    transform_test = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),  # 测试集也要放大
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    full_data = datasets.CIFAR10(root='./data', train=True, download=False)
    train_idx, val_idx = random_split(range(len(full_data)), [45000, 5000])

    train_ds = Subset(datasets.CIFAR10('./data', train=True, transform=transform_train), train_idx.indices)
    val_ds = Subset(datasets.CIFAR10('./data', train=True, transform=transform_test), val_idx.indices)
    test_ds = datasets.CIFAR10('./data', train=False, transform=transform_test)

    # EfficientNet B2 显存占用较大，建议 Batch Size 设置为 64 或 128
    # 如果 128 爆显存，请改为 64
    BATCH_SIZE = 128
    workers = 4 if device.type == 'cuda' else 0

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 2. 初始化 EfficientNet-B2
    print("\n🔨 初始化 EfficientNet-B2 ...")
    model = EfficientNetTransfer(num_classes=10).to(device)

    # 3. 阶段 1: 冻结训练 (Warmup)
    print("\n>>> 阶段 1: 冻结特征层 (Warmup Classifier)")
    model.freeze_layers(freeze=True)
    # AdamW 非常适合 EfficientNet
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    stopper = EarlyStopping(patience=5, path='effnet_b2_best.pth')

    hist1 = train_stage_model(model, train_loader, val_loader, 5, optimizer, scheduler, stopper, "Stage 1")

    # 4. 阶段 2: 全面微调
    print("\n>>> 阶段 2: 解冻全网微调 (Fine-tuning)")
    model.load_state_dict(torch.load('effnet_b2_best.pth'))
    model.freeze_layers(freeze=False)

    # EfficientNet 微调学习率：1e-4 是个黄金值
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    # 150 个 Epochs 足够它收敛
    scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-7)
    stopper = EarlyStopping(patience=20, path='effnet_b2_best.pth')

    hist2 = train_stage_model(model, train_loader, val_loader, 150, optimizer, scheduler, stopper, "Stage 2")

    # 5. 结果
    plot_history(hist1, hist2)

    print("\n🏆 加载最终最佳模型...")
    model.load_state_dict(torch.load('effnet_b2_best.pth'))
    evaluate_test_set(model, test_loader, classes, device)

    torch.save(model.state_dict(), 'final_effnet_b2_95acc.pth')
    print("\n✅ 完成！EfficientNet-B2 模型已保存。")
