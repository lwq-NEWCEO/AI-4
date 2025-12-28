# CIFAR-10 Image Classification: From Baseline to SOTA (93.77%)

## 实验三：基于 CIFAR-10 图像分类模型的设计与实现

作者: lwq-NEWCEO

Github：https://github.com/lwq-NEWCEO/AI-4/

## 我的电脑配置

### 一、硬件配置

处理器：Intel (R) Core (TM) Ultra 9 275HX，24 核，基础主频 2.70GHz

内存：32.0GB 物理内存（可用 31.4GB）

显卡：NVIDIA GeForce RTX 5080 Laptop GPU

### 二、软件配置

操作系统：Windows 11 家庭中文版 64 位

图形接口：DirectX 12

显卡驱动：NVIDIA 573.24 DCH 版

## 📖 1. 项目简介

本项目聚焦于 神经机器翻译任务，旨在通过从零构建并深度调优模型，探索序列生成架构的演进之路。

实验采用“基线复现 -> 架构探索 -> 深度调优 -> 代际对比”的完整闭环策略，在 Multi30k (De-En) 数据集上进行了约 35 次版本迭代。我们从传统的 RNN (Bi-GRU + Attention) 起步，逐步突破性能瓶颈，最终手写实现了 Transformer 架构，并通过 Pre-Norm、权重绑定等现代化策略，将模型性能推向 SOTA 水平。

本项目不仅追求高 BLEU 分数，更致力于通过严谨的控制变量实验，揭示不同架构在参数效率、梯度流动及推理延迟上的本质差异。

核心结论：

“RNN 的深度是梯度流动的壁垒，而 Transformer 的并行计算打破了时序的枷锁。在同等参数量级下，自注意力机制在长程依赖捕捉与推理效率上实现了双重‘代际’超越。”

## 📂 2. 项目结构 
(注：请在后续补充具体的项目目录结构图)

[待补充：项目文件树状图]

## ⚙️ 3. 安装与环境 

本项目基于 Python 3.x 与 PyTorch 构建，强烈建议使用 NVIDIA GPU 进行加速。

主要依赖：

PyTorch

TorchText

SpaCy (用于德语/英语分词)

Matplotlib (用于可视化分析)

环境配置命令：

# 克隆仓库
git clone https://github.com/YourUsername/Seq2Seq-Evolution.git
cd Seq2Seq-Evolution

# 安装依赖
pip install -r requirements.txt

# 下载 SpaCy 语言模型
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

🚀 4. 快速开始 (Usage)

(注：请在后续补充具体的运行脚本和参数说明)

# [待补充：训练与测试的运行命令]

## 📊 5. 实验演进与结果 (Evolution & Results)
### 阶段一：RNN 进化论 (The RNN Evolution)

架构起点： 双向 LSTM (Bi-LSTM)。

优化路径：

轻量化： 替换 LSTM 为 GRU，参数量减少 25%，加速收敛。

结构搜索： 实验发现 E2-D2 (2层编码-2层解码) 为最优深度配置，深层网络 (E4-D4) 因梯度消失导致性能崩塌。

信息瓶颈突破： 引入 Bahdanau Attention，实现源句信息的动态对齐。

推理增强： 使用 Beam Search (k=5) 替代贪婪搜索。

最终表现： BLEU-4 分数从基线的 23.23 提升至 36.00。

### 阶段二：Transformer 的重构与挑战

架构策略： 不使用 nn.Transformer 接口，而是手动搭建 EncoderLayer、MultiHeadAttention 等核心组件，实现“白盒”掌控。

关键问题修复： 修正 Padding Mask 与 Look-ahead Mask 逻辑，解决信息泄露导致的 BLEU 0 问题。

SOTA 调优组合拳：

架构革新： Post-Norm 
→
→ Pre-Norm，将 LayerNorm 移至残差路径上，显著稳定了梯度流。

参数效率： 启用 Weight Tying (共享 Embedding 与 Output 层权重)，减少 1.5M 参数并抑制过拟合。

训练策略： 引入 OneCycleLR 预热策略与 Label Smoothing (0.1)。

最终表现： BLEU-4 达到 40.75，在同参数量下全面超越 RNN。

## 🔍 6. 深入分析 (Deep Analysis)
### 6.1 架构对比：RNN vs Transformer

在严格控制参数量（约 5M-6M）的前提下，我们对两种架构进行了多维度的对比：

维度	RNN (Best: E2-D2 + Attn)	Transformer (Best: Macro-4H)	差异分析
翻译精度 (BLEU-4)	34.87	40.75	+16.9%，Transformer 在长难句表现更好
语义理解 (BERTScore)	60.08	84.52	代际级差距，Transformer 生成的语义向量更准确
训练稳定性 (PPL)	13.39	4.82	Pre-Norm 架构带来了极低的困惑度
推理延迟 (Latency)	90.54ms	79.63ms	-12%，并行计算优势抵消了计算量的增加
### 6.2 关键问题探讨 (Q&A)

#### Q1: 为什么 RNN 模型出现了“越深越差”的退化现象？

实验数据显示，当 RNN 堆叠至 4 层时性能急剧下降。这是因为 RNN 是时间维度的深层网络，梯度需反向传播 
T×L 步，极易引发梯度消失。且在未引入残差连接的情况下，深层信息传递损耗严重。

#### Q2: 为什么计算量更大的 Transformer 推理速度反而更快？

串行阻塞 vs 并行吞吐。RNN 计算 $h_t$ 必须等待 $h_{t-1}$，导致 GPU 核心大量闲置。而 Transformer 虽然总 FLOPs 更高，但能一次性并行处理整个序列，瞬间占满 GPU 核心，大幅减少了 I/O 等待和内核启动开销。

#### Q3: Attention Heads 越多越好吗？

否。实验表明 4 Heads (Macro) 优于 16 Heads (Micro)。在模型维度 
$d_{model} = 256$ 下，16 个头导致每个子空间仅 16 维，特征过于破碎；而 4 个头（64维/头）保留了更完整的语义特征。

## 📝 7. 结论 (Conclusion)

本研究证明，在中小规模数据集上，“Pre-Norm Transformer + 动态学习率调度 + 强正则化” 是最优的工程实践方案。

RNN 的谢幕： 尽管引入 Bi-GRU 和 Attention 能显著提升性能，但在处理长距离依赖和并行计算上存在先天缺陷。

Transformer 的统治： 通过“手术刀”般的架构调优（Pre-Norm, Weight Tying），Transformer 不仅在字面精度上碾压对手，更在语义建模能力上展现了强大的泛化性。

这是一次从代码复现到深度理解的飞跃，验证了 Self-Attention 机制在现代 NLP 中的基石地位。

## 📚 8. 参考文献 (References)

### Part 1: RNN & Variants

Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation

Cho, K., et al. (2014)

https://arxiv.org/abs/1406.1078

Neural Machine Translation by Jointly Learning to Align and Translate

Bahdanau, D., Cho, K., & Bengio, Y. (2014)

https://arxiv.org/abs/1409.0473

Bidirectional Recurrent Neural Networks

Schuster, M., & Paliwal, K. K. (1997)

https://ieeexplore.ieee.org/document/650093

Sequence to Sequence Learning with Neural Networks

Sutskever, I., Vinyals, O., & Le, Q. V. (2014)

https://arxiv.org/abs/1409.3215

### Part 2: Transformer & Architecture Optimization
5. Attention Is All You Need
* Vaswani, A., et al. (2017)
* https://arxiv.org/abs/1706.03762

6. Using the Output Embedding to Improve Language Models
* Press, O., & Wolf, L. (2017)
* https://arxiv.org/abs/1608.05859

7. On Layer Normalization in the Transformer Architecture
* Xiong, R., et al. (2020)
* https://arxiv.org/abs/2002.04745

### Part 3: Training Strategies
8. A Disciplined Approach to Neural Network Hyper-Parameters: Part 1 -- Learning Rate
* Smith, L. N. (2018)
* https://arxiv.org/abs/1803.09820

9. Rethinking the Inception Architecture for Computer Vision
* Szegedy, C., et al. (2016)
* https://arxiv.org/abs/1512.00567

10. Adam: A Method for Stochastic Optimization
* Kingma, D. P., & Ba, J. (2014)
* https://arxiv.org/abs/1412.6980

## 🤝 致谢 (Acknowledgements)

感谢 TorchText 与 SpaCy 社区提供的基础工具支持。实验设计参考了 "Attention Is All You Need" 原文及 PyTorch 官方最佳实践。
