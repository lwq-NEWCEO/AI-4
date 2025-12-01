
## 实验三：基于 CIFAR-10 图像分类模型的设计与实现

作者: 罗文琦 (ID: 10235501417)

## 我的电脑配置

### 一、硬件配置
处理器：Intel (R) Core (TM) Ultra 9 275HX，24 核，基础主频 2.70GHz
内存：32.0GB 物理内存（可用 31.4GB）
显卡：NVIDIA GeForce RTX 5080 Laptop GPU
### 二、软件配置
操作系统：Windows 11 家庭中文版 64 位
图形接口：DirectX 12
显卡驱动：NVIDIA 573.24 DCH 版

## 📖 1. 项目简介 (Overview)
本项目旨在构建一个高效的 CIFAR-10 图像分类系统。实验采用层层递进的策略，历经四个阶段的迭代（基准模型 -> 调优模型 -> 架构重构 -> 迁移学习 SOTA），最终实现了测试集 93.77% 的高准确率。

本项目不仅关注准确率的提升，还深入探讨了模型的可解释性（Grad-CAM）、错误样本分析以及各超参数对性能的贡献（消融实验）。

核心结论：

“迁移学习决定下限，分辨率决定上限，数据增强决定细节。”

## 📂 2. 项目结构 (Project Structure)
```text
CIFAR10-Classification/
├── data/                  # CIFAR-10 数据集 (自动下载)
├── models/                # 模型定义 (SimpleCNN, RefinedCNN, EfficientNetTransfer)
├── checkpoints/           # 训练好的模型权重 (.pth)
├── results/               # 实验结果图表 (Loss曲线, 混淆矩阵, Grad-CAM)
├── scripts/               # 核心执行脚本
│   ├── train_baseline.py  # 阶段一 & 二：基准与调优训练
│   ├── train_sota.py      # 阶段三 & 四：EfficientNet 训练与消融实验
│   ├── visualize.py       # Grad-CAM 与 错误样本分析可视化
│   └── utils.py           # 工具函数 (早停, 随机种子, 绘图)
├── requirements.txt       # 项目依赖
└── README.md              # 项目说明文档
```
## ⚙️ 3. 安装与环境 (Installation)
本项目基于 Python 3.14 开发，支持 CPU 和 GPU 训练（推荐使用 GPU）。
```bash
git clone https://github.com/YourUsername/CIFAR10-Project.git
cd CIFAR10-Project
```

安装依赖
```
pip install -r requirements.txt
```

硬件检查

代码会自动检测计算设备：

GPU: 启用 CUDA 加速 (推荐显存 >= 4GB)

CPU: 可运行，但训练速度较慢

## 🚀 4. 快速开始 (Usage)
运行 SOTA 模型训练 (阶段三 Pro)
直接复现最高准确率 (93.77%) 的实验结果：

```bash
python Adapt-Attention-3pro.py
```
运行可视化分析
加载训练好的模型，生成 Grad-CAM 热力图和混淆矩阵：

```bash

python Plot.py
```

运行消融实验

一键运行 5 组对比实验（需耗时较长）：

```bash

python MVP.py
```

## 📊 5. 实验演进与结果 (Experiment Pipeline)

### 阶段一：基准模型 (SimpleCNN)

架构: 3层简单卷积网络。

表现: 训练集 85.03% | 验证集 73.30%。

问题: 严重过拟合，模型在 Epoch 35 后遇到性能瓶颈。

### 阶段二：调优 (RefinedCNN)

改进: 引入 BatchNorm2d，加入 L2 正则化 (weight_decay=5e-4)，使用 CosineAnnealingLR 调度器。

数据增强: RandomCrop + RandomHorizontalFlip + ColorJitter。

表现: 验证集 84.72%。

分析: 有效缓解了过拟合，Loss 曲线平滑收敛，但浅层网络到达天花板。

### 阶段三 Pro：SOTA 突破 (EfficientNet-B2)

策略: 迁移学习 (ImageNet 预训练) + 分辨率适配。

关键优化:

Resize 64x64: 解决了 32x32 图片经 ResNet/EfficientNet 下采样后特征图变 1x1 丢失空间信息的问题。

两阶段训练: Warmup (冻结特征层) -> Fine-tuning (全解冻)。

强正则化: RandomErasing + Label Smoothing + Dropout(0.3)。

表现: 测试集 93.77%。

## 🔍 6. 深入分析 (Analysis & Visualization)

### 6.1 消融实验 (Ablation Study)

通过控制变量法，量化了各模块的贡献：

实验组	准确率	贡献度	结论
基准 (EfficientNet-B2)	93.81%	-	SOTA 表现
无预训练 (No Pretrain)	83.74%	-10.07%	决定下限：预训练权重至关重要
无分辨率放大 (No Resize)	85.99%	-7.82%	决定上限：解决空间特征丢失问题
无随机擦除 (No RandomErasing)	92.93%	-0.88%	决定细节：提升鲁棒性和泛化能力

### 6.2 可视化解释 (Grad-CAM)

Bird: 热力图精准聚焦于鸟的躯干，忽略了复杂的树枝背景。

Truck: 热力图横向分布，关注底盘和轮子。

Plane/Ship: 模型在一定程度上依赖背景颜色（如蓝色背景）进行辅助判断。

### 6.3 错误案例分析

Cat vs Dog: 极度相似的生物特征（如猫头鹰的面部被误判为狗）。

Contextual Bias: 红色卡车因大面积蓝色背景被误判为飞机。

## 📝 7. 结论 (Conclusion)
通过本次实验，我们证明了在 CIFAR-10 等小尺寸数据集上，“预训练权重 + 分辨率上采样适配 + 强正则化策略” 是最优的工程实践方案。

迁移学习提供了通用的特征表达。

分辨率适配 (64x64) 使得深层网络能保留空间特征。

Random Erasing 进一步挖掘了模型的泛化潜力。


## 🤝 8. 致谢 (Acknowledgments)

感谢 PyTorch 团队提供的优秀深度学习框架。

感谢 pytorch-grad-cam 库提供的可视化支持。

实验设计参考了 EfficientNet 论文及相关最佳实践。
