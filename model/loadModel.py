# model/loadModel.py
# 任务 1.2：构建 ResNet50 批量推理基准
# 功能：加载预训练 ResNet50，遍历 testset 目录，输出 Top-1 ~ Top-5 分类结果与置信度

import os
import sys
import json
import urllib.request
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ==================== 1. 硬件环境适配 ====================
# 自动检测 CUDA 是否可用，优先挂载 NVIDIA RTX 5060 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 当前计算设备: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU 型号: {torch.cuda.get_device_name(0)}")


# ==================== 2. 模型与权重加载 ====================
# 使用官方推荐写法 weights=ResNet50_Weights.DEFAULT 加载 IMAGENET1K_V2 预训练权重
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = model.to(device)          # 将模型挂载至 GPU/CPU
model.eval()                      # 冻结网络层，进入推理评估模式（关闭 Dropout/BN 更新）
print("[INFO] ResNet50 模型加载完成，已切换至 eval 模式。")


# ==================== 3. 数据预处理 ====================
# 调用与当前权重严格匹配的官方预处理管道（Resize, CenterCrop, Normalize）
preprocess = weights.transforms()


# ==================== 4. 标签映射 ====================
# 动态抓取 ImageNet-1K 类别标签 JSON，若失败则回退为 Class ID
IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/"
    "master/imagenet-simple-labels.json"
)
labels = []
try:
    with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=10) as response:
        labels = json.loads(response.read().decode("utf-8"))
    print(f"[INFO] 成功从网络加载 {len(labels)} 个 ImageNet 类别标签。")
except Exception as e:
    print(f"[WARN] 无法获取远程标签映射: {e}")
    print("[WARN] 将使用 Class ID 作为标签回退。")
    labels = [str(i) for i in range(1000)]


def get_label(class_id: int) -> str:
    """根据类别索引返回可读标签，若越界则返回 ID 字符串。"""
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return str(class_id)


# ==================== 5. 批量处理逻辑 ====================
# 确定 testset 目录路径（与脚本同级的 testset 文件夹）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTSET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "testset")

# 支持的图像后缀
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def infer_single_image(image_path: str):
    """
    对单张图片执行前向推理，返回 (top5_probs, top5_classes)。
    所有张量计算均在 with torch.no_grad() 上下文中执行以节省显存。
    """
    # 打开图像并转为 RGB 模式（部分 PNG 含透明度通道）
    raw_image = Image.open(image_path).convert("RGB")

    # 应用官方预处理并增加 Batch 维度: (C, H, W) -> (1, C, H, W)
    input_tensor = preprocess(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():  # 禁用梯度计算，降低显存占用并加速推理
        outputs = model(input_tensor)  # 前向传播，输出 (1, 1000) 的 logits

    # 使用 Softmax 将 logits 转换为概率分布
    probabilities = F.softmax(outputs, dim=1)

    # 提取 Top-5 概率值与对应类别索引
    top5_probs, top5_classes = torch.topk(probabilities, k=5, dim=1)

    # 将结果移回 CPU 并转为 Python 原生类型
    top5_probs = top5_probs.squeeze(0).cpu().numpy()
    top5_classes = top5_classes.squeeze(0).cpu().numpy()

    return top5_probs, top5_classes


def main():
    # 鲁棒性：检查 testset 目录是否存在
    if not os.path.isdir(TESTSET_DIR):
        print(f"[ERROR] 测试目录不存在: {TESTSET_DIR}")
        sys.exit(1)

    # 遍历目录并过滤图像文件
    image_files = [
        f for f in os.listdir(TESTSET_DIR)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]
    image_files.sort()

    if not image_files:
        print(f"[WARN] 在 {TESTSET_DIR} 中未找到支持的图像文件。")
        sys.exit(0)

    success_count = 0

    print("\n" + "=" * 60)
    print("开始批量推理...")
    print("=" * 60 + "\n")

    for filename in image_files:
        file_path = os.path.join(TESTSET_DIR, filename)

        try:
            top5_probs, top5_classes = infer_single_image(file_path)
        except Exception as e:
            # 鲁棒性：单张图片损坏时跳过，防止整体崩溃
            print(f"[SKIP] 读取失败: {filename} ({e})")
            continue

        # 终端输出格式规范：Top-1 主要结果
        top1_class = top5_classes[0]
        top1_prob = top5_probs[0] * 100
        top1_label = get_label(int(top1_class))

        print(f"“{filename}” 【{top1_label}】 {top1_prob:.2f}%")

        # 额外输出 Top-2 ~ Top-5 结果（缩进展示）
        for rank in range(1, 5):
            cls_id = int(top5_classes[rank])
            prob = top5_probs[rank] * 100
            label = get_label(cls_id)
            print(f"    Top-{rank + 1}: 【{label}】 {prob:.2f}%")

        success_count += 1
        print()  # 空行分隔每张图片结果

    print("=" * 60)
    print(f"推理完成。成功识别图片总数: {success_count} / {len(image_files)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
