import os
import warnings

warnings.filterwarnings('ignore')  # 忽略无关警告
# 解决OpenMP冲突+GPU内存优化
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from torchvision.models.resnet import BasicBlock, ResNet
from tqdm import tqdm  # 进度条可视化
import copy  # 保存最佳模型


# --- 1. 核心工具函数：随机种子+早停机制（防止过拟合） ---
def set_seed(seed=42):
    """固定全链路随机种子，确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # 关闭CuDNN自动优化，进一步保证稳定性


class EarlyStopping:
    """早停机制：验证损失连续不下降则停止训练，保存最佳模型"""

    def __init__(self, patience=10, min_delta=1e-4, path='best_model.pth'):
        self.patience = patience  # 容忍多少个epoch无提升
        self.min_delta = min_delta  # 最小提升阈值
        self.path = path  # 最佳模型保存路径
        self.counter = 0  # 无提升计数器
        self.best_loss = float('inf')  # 最佳验证损失
        self.early_stop = False  # 是否早停标志

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            # 验证损失下降，更新最佳损失并保存模型
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f"✅ 验证损失下降至 {val_loss:.4f}，保存最佳模型")
        else:
            # 验证损失无提升，计数器+1
            self.counter += 1
            print(f"⚠️  早停计数器: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"🚫 触发早停！最佳验证损失: {self.best_loss:.4f}")


# --- 2. 注意力机制：SE Block（轻量高效，适配ResNet） ---
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block：通道注意力机制"""

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze：全局平均池化（压缩空间维度）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation：全连接层（学习通道权重）
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 升维
            nn.Sigmoid()  # 输出0-1的权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 全局平均池化：(b,c,h,w) → (b,c,1,1) → (b,c)
        y = self.avg_pool(x).view(b, c)
        # 学习通道权重：(b,c) → (b,c)
        y = self.fc(y).view(b, c, 1, 1)
        # 特征加权：逐通道相乘
        return x * y.expand_as(x)


# --- 3. 改进ResNet18：集成SE Block（替换原有BasicBlock） ---
class SEBasicBlock(BasicBlock):
    """带SE注意力的ResNet基础块"""

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBasicBlock, self).__init__(
            inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer
        )
        # 在Block末尾添加SE Block
        self.se = SEBlock(planes, reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 插入SE注意力加权
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


# 构建SE-ResNet18
def se_resnet18(pretrained=False, num_classes=1000, reduction=16):
    """带SE注意力的ResNet18模型"""
    norm_layer = nn.BatchNorm2d
    model = ResNet(
        SEBasicBlock,  # 用SEBasicBlock替换默认BasicBlock
        [2, 2, 2, 2],  # ResNet18的Block数量
        num_classes=num_classes,
        norm_layer=norm_layer
    )
    model.inplanes = 64
    model.dilation = 1
    model.base_width = 64
    model.groups = 1

    # 初始化权重
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # 如果需要预训练权重（这里我们后续加载官方ResNet18预训练权重，再适配SE层）
    if pretrained:
        # 加载官方ResNet18预训练权重
        resnet18_pretrained = models.resnet18(pretrained=True)
        # 复制除最后全连接层外的权重（SE层权重会随机初始化）
        pretrained_dict = resnet18_pretrained.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('fc')}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


# --- 4. 迁移学习模型封装（SE-ResNet18） ---
class SETransferModel(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(SETransferModel, self).__init__()
        # 加载带SE注意力的ResNet18（预训练权重适配）
        self.base_model = se_resnet18(pretrained=pretrained, num_classes=num_classes)
        # 替换最后全连接层（确保输出适配CIFAR-10的10类）
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.3),  # 添加Dropout抑制过拟合
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

    def freeze_layers(self, freeze=True):
        """冻结/解冻卷积层（仅训练/微调分类头）"""
        for param in self.base_model.parameters():
            param.requires_grad = not freeze
        # 确保分类头始终可训练
        for param in self.base_model.fc.parameters():
            param.requires_grad = True


# --- 5. 优化后的训练函数（分阶段微调+AdamW+早停） ---
def train_stage_model(model, train_loader, val_loader, epochs, optimizer, scheduler, early_stopping, stage_name):
    """分阶段训练函数（适配两阶段微调）"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑，进一步抑制过拟合
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0.0

    print(f"\n===== 开始 {stage_name}（{epochs}个Epoch）=====")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 进度条可视化
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Train)")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计训练指标
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 更新进度条
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        train_bar.close()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} (Val)")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 统计验证指标
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            val_bar.close()

        # 计算平均指标
        avg_train_loss = train_loss / train_total
        avg_train_acc = 100 * train_correct / train_total
        avg_val_loss = val_loss / val_total
        avg_val_acc = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        # 更新历史记录
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        history['lr'].append(current_lr)

        # 更新最佳验证准确率
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc

        # 打印日志
        print(f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}% | "
              f"Best Val Acc: {best_val_acc:.2f}%")

        # 早停检查
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            break

        # 更新学习率
        scheduler.step()

    print(f"===== {stage_name}结束 =====")
    return history


# --- 6. 测试集评估函数（含类别级准确率） ---
def evaluate_test_set(model, test_loader, classes, device):
    """详细评估测试集：整体准确率+每个类别的准确率"""
    model.eval()
    test_correct = 0
    test_total = 0
    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing")
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # 整体统计
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # 类别级统计
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # 计算整体准确率
    overall_acc = 100 * test_correct / test_total
    print(f"\n📊 测试集整体准确率: {overall_acc:.2f}%")

    # 计算类别级准确率
    print("\n类别级准确率：")
    for i in range(len(classes)):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"  {classes[i]:<10}: {class_acc:.2f}%")
        else:
            print(f"  {classes[i]:<10}: 无数据")

    return overall_acc, class_correct, class_total


# --- 7. 可视化函数（对比训练曲线） ---
def plot_combined_history(history_stage1, history_stage2, save_path='se_resnet18_training_history.png'):
    """合并两阶段训练历史，绘制综合曲线"""
    # 合并两阶段数据
    total_train_loss = history_stage1['train_loss'] + history_stage2['train_loss']
    total_train_acc = history_stage1['train_acc'] + history_stage2['train_acc']
    total_val_loss = history_stage1['val_loss'] + history_stage2['val_loss']
    total_val_acc = history_stage1['val_acc'] + history_stage2['val_acc']
    total_lr = history_stage1['lr'] + history_stage2['lr']
    total_epochs = len(total_train_loss)

    # 创建图表
    plt.figure(figsize=(18, 5))

    # 1. 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(total_train_loss, label='Train Loss', color='#1f77b4', linewidth=1.5)
    plt.plot(total_val_loss, label='Val Loss', color='#ff7f0e', linewidth=1.5)
    plt.axvline(x=len(history_stage1['train_loss']) - 1, color='red', linestyle='--', alpha=0.7, label='Stage 1 End')
    plt.title('Loss History (SE-ResNet18)', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    # 2. 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(total_train_acc, label='Train Accuracy', color='#1f77b4', linewidth=1.5)
    plt.plot(total_val_acc, label='Val Accuracy', color='#ff7f0e', linewidth=1.5)
    plt.axvline(x=len(history_stage1['train_acc']) - 1, color='red', linestyle='--', alpha=0.7, label='Stage 1 End')
    plt.title('Accuracy History (SE-ResNet18)', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(alpha=0.3)

    # 3. 学习率曲线
    plt.subplot(1, 3, 3)
    plt.plot(total_lr, label='Learning Rate', color='#2ca02c', linewidth=1.5)
    plt.axvline(x=len(history_stage1['lr']) - 1, color='red', linestyle='--', alpha=0.7, label='Stage 1 End')
    plt.title('Learning Rate Schedule', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n训练曲线已保存至: {save_path}")


# --- 主程序入口 ---
if __name__ == '__main__':
    # 1. 基础配置（最优参数适配）
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 2. 数据预处理（增强策略优化：更适配ResNet）
    # 训练集增强（保留有效策略，新增随机旋转）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪+填充
        transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转±15°
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 温和颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.2)  # 随机擦除（模拟遮挡，提升鲁棒性）
    ])

    # 验证集/测试集：仅标准化（无增强）
    transform_test_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    # 3. 数据集加载与划分（45k训练+5k验证）
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=None)
    train_indices, val_indices = random_split(range(len(full_train_dataset)), [45000, 5000])

    train_dataset = Subset(
        datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train),
        train_indices.indices
    )
    val_dataset = Subset(
        datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test_val),
        val_indices.indices
    )
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test_val)

    # 4. DataLoader配置（根据GPU内存调整Batch Size）
    BATCH_SIZE = 256  # 最优Batch Size（GPU内存≥8G推荐，否则改为128/64）
    NUM_WORKERS = 4 if device.type == 'cuda' else 0  # GPU用4进程，CPU用0进程

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True  # pin_memory加速GPU数据传输，drop_last避免批次不一致
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 5. 模型初始化
    model = SETransferModel(num_classes=len(classes), pretrained=True).to(device)
    print(f"\n📦 模型结构：SE-ResNet18（带通道注意力机制）")
    print(f"📦 模型参数总数：{sum(p.numel() for p in model.parameters()):,}")
    print(f"📦 可训练参数总数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 6. 两阶段训练配置（最优迁移学习策略）
    # 阶段1：冻结卷积层，仅训练分类头（5个Epoch，快速收敛分类头）
    model.freeze_layers(freeze=True)
    optimizer_stage1 = optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=5e-4  # AdamW+L2正则
    )
    scheduler_stage1 = CosineAnnealingLR(optimizer_stage1, T_max=5, eta_min=1e-5)
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4, path='se_resnet18_best.pth')

    # 阶段1训练
    history_stage1 = train_stage_model(
        model, train_loader, val_loader, epochs=5,
        optimizer=optimizer_stage1, scheduler=scheduler_stage1,
        early_stopping=early_stopping, stage_name="阶段1：冻结卷积层训练分类头"
    )

    # 阶段2：解冻所有层，微调整个模型（95个Epoch，适配CIFAR-10）
    model.freeze_layers(freeze=False)  # 解冻所有层
    optimizer_stage2 = optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=5e-4  # 更小的学习率，避免破坏预训练特征
    )
    scheduler_stage2 = CosineAnnealingLR(optimizer_stage2, T_max=95, eta_min=1e-6)

    # 阶段2训练（重置早停计数器）
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4, path='se_resnet18_best.pth')
    history_stage2 = train_stage_model(
        model, train_loader, val_loader, epochs=95,
        optimizer=optimizer_stage2, scheduler=scheduler_stage2,
        early_stopping=early_stopping, stage_name="阶段2：解冻所有层微调"
    )

    # 7. 加载最佳模型并评估测试集
    print(f"\n🔍 加载最佳模型进行测试集评估...")
    model.load_state_dict(torch.load('se_resnet18_best.pth'))
    model.to(device)
    test_acc, class_correct, class_total = evaluate_test_set(model, test_loader, classes, device)

    # 8. 可视化训练曲线
    plot_combined_history(history_stage1, history_stage2)

    # 9. 保存最终模型（含结构+权重）
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'classes': classes
    }, 'se_resnet18_final.pth')
    print(f"\n💾 最终模型已保存至: se_resnet18_final.pth")