import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# --- 确保引入了之前的模型定义 ---
# 如果是在同一个脚本里，不需要重新复制 EfficientNetTransfer 类
# 如果是新脚本，请把 EfficientNetTransfer 类定义粘贴在这里
# from your_script import EfficientNetTransfer

def inverse_normalize(tensor, mean, std):
    """反标准化，用于将 Tensor 转回图片显示"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def visualize_gradcam(model_path, device, num_images=5):
    print(f"🔍 正在加载模型用于 Grad-CAM 可视化: {model_path} ...")

    # 1. 加载模型
    model = EfficientNetTransfer(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 定义目标层 (Target Layer)
    # 对于 EfficientNet，我们通常关注最后一个卷积层，它包含了最丰富的高级语义特征
    # 在 torchvision 的实现中，它通常位于 .features 的最后一块
    target_layers = [model.base_model.features[-1]]

    # 3. 数据准备 (必须和训练时的一致: Resize 64x64)
    # 注意：这里我们需要两个 transform
    # transform_input: 给模型看的 (含 Normalize)
    # transform_display: 给人类和 GradCAM 绘图用的 (不含 Normalize，只转 Tensor)
    RESIZE_SIZE = 64
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform_input = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 加载测试集
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_input)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 4. 初始化 Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)  # use_cuda=True if device=='cuda'

    # 5. 随机抽取图片进行可视化
    indices = np.random.choice(len(test_ds), num_images, replace=False)

    plt.figure(figsize=(15, 3 * num_images))

    for i, idx in enumerate(indices):
        input_tensor, label_id = test_ds[idx]
        input_tensor = input_tensor.unsqueeze(0).to(device)  # 增加 batch 维度: [1, 3, 64, 64]

        # --- 获取模型预测结果 ---
        output = model(input_tensor)
        _, predicted_id = torch.max(output, 1)
        predicted_label = classes[predicted_id.item()]
        true_label = classes[label_id]

        # --- 运行 Grad-CAM ---
        # targets=None 表示自动寻找置信度最高的类别（即模型的预测结果）
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)

        # 在这张图中，grayscale_cam 是 [1, 64, 64]
        grayscale_cam = grayscale_cam[0, :]

        # --- 准备背景图 ---
        # 我们需要反标准化回去，变成 0-1 之间的 float 用于显示
        rgb_img = inverse_normalize(input_tensor.cpu().squeeze(0).clone(), mean, std)
        rgb_img = rgb_img.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        rgb_img = np.clip(rgb_img, 0, 1)  # 限制在 0-1 之间

        # 将热力图叠加到原图上
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # --- 绘图 ---
        # 1. 原图
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(rgb_img)
        plt.title(f"Original: {true_label}")
        plt.axis('off')

        # 2. 纯热力图
        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(grayscale_cam, cmap='jet')
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')

        # 3. 叠加图
        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(visualization)
        # 如果预测正确用绿色，错误用红色
        color = 'green' if predicted_id == label_id else 'red'
        plt.title(f"Pred: {predicted_label}", color=color, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=300)
    print("\n✅ Grad-CAM 可视化完成！图片已保存为 'gradcam_visualization.png'")
    plt.show()


# --- 运行部分 ---
if __name__ == '__main__':
    # 确保你有这个文件，或者改成你自己保存的模型路径
    model_path = 'final_EfficientNetB2_Pretrained.pth'

    # 检查文件是否存在，防止报错
    import os

    if not os.path.exists(model_path):
        # 尝试使用你上一个代码块可能保存的名字
        model_path = 'final_effnet_b2.pth'

    if os.path.exists(model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        visualize_gradcam(model_path, device, num_images=5)
    else:
        print(f"❌ 未找到模型文件: {model_path}，请检查路径。")
