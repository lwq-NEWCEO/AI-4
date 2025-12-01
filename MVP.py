import os
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import time  # 导入 time 模块用于计时

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
    def __init__(self, num_classes=10, load_pretrained=True):  # 增加 load_pretrained 参数
        super(EfficientNetTransfer, self).__init__()

        print(f"🔄 正在加载 EfficientNet-B2 {'预训练' if load_pretrained else '随机初始化'} 权重...")

        if load_pretrained:
            if EfficientNet_B2_Weights is not None:
                weights = EfficientNet_B2_Weights.DEFAULT
                self.base_model = efficientnet_b2(weights=weights)
            else:
                self.base_model = efficientnet_b2(pretrained=True)
        else:  # 不加载预训练，随机初始化
            self.base_model = efficientnet_b2(pretrained=False)  # EfficientNet 默认是随机初始化

        original_fc = self.base_model.classifier[1]
        in_features = original_fc.in_features

        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

    def freeze_layers(self, freeze=True):
        for param in self.base_model.parameters():
            param.requires_grad = not freeze

        for param in self.base_model.classifier.parameters():
            param.requires_grad = True


# --- 3. 训练流程函数 (不变) ---
def train_stage_model(model, train_loader, val_loader, epochs, optimizer, scheduler, early_stopping, stage_name):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0.0

    print(f"\n===== 开始 {stage_name}（共 {epochs} Epochs）=====")

    for epoch in range(epochs):
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


# --- 4. 评估流程函数 (添加显存和推理时间测量) ---
def evaluate_test_set(model, test_loader, classes, device):
    model.eval()
    test_correct = 0
    test_total = 0
    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))

    # 显存使用量
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        start_mem = torch.cuda.memory_allocated(device)

    start_time = time.time()

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

    end_time = time.time()
    inference_time = (end_time - start_time) / test_total * 1000  # 单张图片推理时间 (ms)
    fps = test_total / (end_time - start_time)  # FPS

    acc = 100 * test_correct / test_total
    print(f"\n📊 测试集最终准确率: {acc:.2f}%")
    print("-" * 30)
    for i in range(len(classes)):
        if class_total[i] > 0:
            print(f"  {classes[i]:<10}: {100 * class_correct[i] / class_total[i]:.2f}%")
    print("-" * 30)
    print(f"⏱️ 平均单张图片推理时间: {inference_time:.2f} ms")
    print(f"⚡ 推理速度: {fps:.2f} FPS")

    if device.type == 'cuda':
        end_mem = torch.cuda.memory_allocated(device)
        peak_mem = torch.cuda.max_memory_allocated(device)
        print(f"📈 GPU显存占用 (MB): {peak_mem / (1024 ** 2):.2f} (峰值)")
        return acc, peak_mem / (1024 ** 2), inference_time

    return acc, None, inference_time


# --- 5. 绘图函数 (不变) ---
def plot_history(hist1, hist2, filename='effnet_b2_history.png', title_suffix=""):
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
    plt.title(f'Loss {title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.axvline(x=len(hist1['train_acc']) - 1, color='r', linestyle='--', alpha=0.5, label='Stage 1 End')
    plt.title(f'Accuracy {title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(lr, label='LR', color='green')
    plt.axvline(x=len(hist1['lr']) - 1, color='r', linestyle='--', alpha=0.5, label='Stage 1 End')
    plt.title(f'Learning Rate {title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\n📈 曲线图已保存至 {filename}")


# --- 主程序 ---
def run_experiment(model_type, pretrained, resize_factor, label_smoothing, random_erasing_p, base_path="."):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"\n--- 实验配置: {model_type} | Pretrained: {pretrained} | Resize: {resize_factor}x | LS: {label_smoothing} | RE_p: {random_erasing_p} ---")
    print(f"🚀 Device: {device}")

    # 1. 数据准备
    RESIZE_SIZE = 32 * resize_factor  # 动态调整分辨率

    transform_train = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomCrop(RESIZE_SIZE, padding=8 if RESIZE_SIZE > 32 else 4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=random_erasing_p)  # 动态调整 RandomErasing
    ])

    transform_test = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    full_data = datasets.CIFAR10(root=os.path.join(base_path, 'data'), train=True,
                                 download=True)  # download=True 确保数据存在
    train_idx, val_idx = random_split(range(len(full_data)), [45000, 5000])

    train_ds = Subset(datasets.CIFAR10(os.path.join(base_path, 'data'), train=True, transform=transform_train),
                      train_idx.indices)
    val_ds = Subset(datasets.CIFAR10(os.path.join(base_path, 'data'), train=True, transform=transform_test),
                    val_idx.indices)
    test_ds = datasets.CIFAR10(os.path.join(base_path, 'data'), train=False, transform=transform_test)

    BATCH_SIZE = 128
    workers = 4 if device.type == 'cuda' else 0

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 2. 初始化 EfficientNet-B2
    print(f"\n🔨 初始化 {model_type} ...")
    model = EfficientNetTransfer(num_classes=10, load_pretrained=pretrained).to(device)

    # 3. 阶段 1: 冻结训练 (Warmup)
    print("\n>>> 阶段 1: 冻结特征层 (Warmup Classifier)")
    model.freeze_layers(freeze=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # CosineAnnealingLR 需要传入当前的 epochs，这里 Stage 1 是 5 epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    stopper = EarlyStopping(patience=5, path=os.path.join(base_path, f'{model_type}_best_s1.pth'))

    hist1 = train_stage_model(model, train_loader, val_loader, 5, optimizer, scheduler, stopper, "Stage 1")

    # 4. 阶段 2: 全面微调
    print("\n>>> 阶段 2: 解冻全网微调 (Fine-tuning)")
    # 只有当 Stage 1 成功运行并保存了模型，才加载
    if not stopper.early_stop:
        model.load_state_dict(torch.load(os.path.join(base_path, f'{model_type}_best_s1.pth')))
    else:  # 如果 Stage 1 就早停了，直接用当前模型状态继续
        print("Stage 1 早停，直接使用当前模型状态进入 Stage 2")

    model.freeze_layers(freeze=False)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    # CosineAnnealingLR 需要传入当前的 epochs，这里 Stage 2 是 150 epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-7)
    stopper = EarlyStopping(patience=20, path=os.path.join(base_path, f'{model_type}_best_s2.pth'))

    hist2 = train_stage_model(model, train_loader, val_loader, 150, optimizer, scheduler, stopper, "Stage 2")

    # 5. 结果
    plot_history(hist1, hist2, filename=os.path.join(base_path, f'{model_type}_history.png'),
                 title_suffix=f"({model_type})")

    print("\n🏆 加载最终最佳模型...")
    # 只有当 Stage 2 成功运行并保存了模型，才加载
    if not stopper.early_stop:
        model.load_state_dict(torch.load(os.path.join(base_path, f'{model_type}_best_s2.pth')))
    else:  # 如果 Stage 2 也早停了，说明当前模型就是最佳
        print("Stage 2 早停，使用当前模型状态进行评估")

    test_acc, peak_mem, inference_time = evaluate_test_set(model, test_loader, classes, device)

    torch.save(model.state_dict(), os.path.join(base_path, f'final_{model_type}.pth'))
    print(f"\n✅ 完成！{model_type} 模型已保存。")

    return test_acc, peak_mem, inference_time, model  # 返回模型对象以计算参数量


if __name__ == '__main__':
    # 确保有地方存放结果
    results_dir = "./experiment_results"
    os.makedirs(results_dir, exist_ok=True)

    experiment_results = {}

    # --- 实验 1: EfficientNet-B2 (Pretrained) ---
    print("\n\n=== 运行实验: EfficientNet-B2 (Pretrained) (基准模型) ===")
    acc_pretrained, mem_pretrained, time_pretrained, model_pretrained = run_experiment(
        "EfficientNetB2_Pretrained", pretrained=True, resize_factor=2, label_smoothing=0.1,
        random_erasing_p=0.2, base_path=results_dir
    )
    param_count_pretrained = sum(p.numel() for p in model_pretrained.parameters() if p.requires_grad)
    experiment_results["EfficientNetB2_Pretrained"] = {
        "Accuracy": acc_pretrained, "Memory(MB)": mem_pretrained, "Inference Time(ms)": time_pretrained,
        "Params": param_count_pretrained
    }

    # --- 实验 2: EfficientNet-B2 (No Pretrain) ---
    print("\n\n=== 运行实验: EfficientNet-B2 (No Pretrain) ===")
    acc_no_pretrain, mem_no_pretrain, time_no_pretrain, model_no_pretrain = run_experiment(
        "EfficientNetB2_NoPretrain", pretrained=False, resize_factor=2, label_smoothing=0.1,
        random_erasing_p=0.2, base_path=results_dir
    )
    param_count_no_pretrain = sum(p.numel() for p in model_no_pretrain.parameters() if p.requires_grad)
    experiment_results["EfficientNetB2_NoPretrain"] = {
        "Accuracy": acc_no_pretrain, "Memory(MB)": mem_no_pretrain, "Inference Time(ms)": time_no_pretrain,
        "Params": param_count_no_pretrain
    }

    # --- 实验 3: EfficientNet-B2 (Pretrained) - No Resize (32x32) ---
    print("\n\n=== 运行实验: EfficientNet-B2 (Pretrained) - No Resize (32x32) ===")
    acc_no_resize, mem_no_resize, time_no_resize, model_no_resize = run_experiment(
        "EfficientNetB2_NoResize", pretrained=True, resize_factor=1, label_smoothing=0.1,
        random_erasing_p=0.2, base_path=results_dir
    )
    param_count_no_resize = sum(p.numel() for p in model_no_resize.parameters() if p.requires_grad)
    experiment_results["EfficientNetB2_NoResize"] = {
        "Accuracy": acc_no_resize, "Memory(MB)": mem_no_resize, "Inference Time(ms)": time_no_resize,
        "Params": param_count_no_resize
    }

    # --- 实验 4: EfficientNet-B2 (Pretrained) - No Label Smoothing ---
    print("\n\n=== 运行实验: EfficientNet-B2 (Pretrained) - No Label Smoothing ===")
    acc_no_ls, mem_no_ls, time_no_ls, model_no_ls = run_experiment(
        "EfficientNetB2_NoLS", pretrained=True, resize_factor=2, label_smoothing=0.0,  # LS设为0
        random_erasing_p=0.2, base_path=results_dir
    )
    param_count_no_ls = sum(p.numel() for p in model_no_ls.parameters() if p.requires_grad)
    experiment_results["EfficientNetB2_NoLS"] = {
        "Accuracy": acc_no_ls, "Memory(MB)": mem_no_ls, "Inference Time(ms)": time_no_ls, "Params": param_count_no_ls
    }

    # --- 实验 5: EfficientNet-B2 (Pretrained) - No Random Erasing ---
    print("\n\n=== 运行实验: EfficientNet-B2 (Pretrained) - No Random Erasing ===")
    acc_no_re, mem_no_re, time_no_re, model_no_re = run_experiment(
        "EfficientNetB2_NoRE", pretrained=True, resize_factor=2, label_smoothing=0.1,
        random_erasing_p=0.0, base_path=results_dir  # RE的p设为0
    )
    param_count_no_re = sum(p.numel() for p in model_no_re.parameters() if p.requires_grad)
    experiment_results["EfficientNetB2_NoRE"] = {
        "Accuracy": acc_no_re, "Memory(MB)": mem_no_re, "Inference Time(ms)": time_no_re, "Params": param_count_no_re
    }

    print("\n\n========= 所有实验结果汇总 =========")
    for exp_name, results in experiment_results.items():
        print(f"--- {exp_name} ---")
        for k, v in results.items():
            if k == "Params":
                print(f"  {k}: {v:,}")
            elif k == "Accuracy":
                print(f"  {k}: {v:.2f}%")
            elif k == "Memory(MB)":
                print(f"  {k}: {v:.2f} MB")
            elif k == "Inference Time(ms)":
                print(f"  {k}: {v:.2f} ms")
            else:
                print(f"  {k}: {v}")

    # 可以将这些结果保存到 CSV 或 JSON 文件中，方便报告生成
    import pandas as pd

    df_results = pd.DataFrame.from_dict(experiment_results, orient='index')
    df_results.index.name = 'Experiment'
    df_results_path = os.path.join(results_dir, "ablation_study_results.csv")
    df_results.to_csv(df_results_path)
    print(f"\n所有实验结果已保存至 {df_results_path}")

