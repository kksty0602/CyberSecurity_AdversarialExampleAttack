# ResNet50-Adversarial-Lab

基于 PyTorch 的 ResNet50 图像识别与目标定向对抗攻击（Targeted Adversarial Attack）研究项目。

## 项目简介

本项目旨在探索和复现深度卷积神经网络（CNN）在计算机视觉任务中的表现及其安全性。项目以官方预训练的 ResNet50 模型为基础，包含两个核心实验模块：

1. **基线推理（Baseline Inference）**：实现对本地图像数据集的高效批量分类与置信度评估。
2. **对抗样本攻击（Adversarial Attack）**：利用梯度下降算法（如 FGSM/PGD）生成微小扰动，实现目标定向攻击（例如：将模型识别高度自信的"犬类"图像，通过像素级干扰，定向欺骗为"母鸡"）。

## 硬件与运行环境

本项目已在较新的硬件架构上完成测试与适配。针对 NVIDIA RTX 50 系列显卡（sm_120 架构），标准稳定版 PyTorch 可能会出现算力不兼容警告，因此推荐使用 Nightly 预览版。

* **操作系统**: Windows
* **计算设备**: NVIDIA GeForce RTX 5060 Laptop GPU
* **编程语言**: Python 3.13
* **深度学习框架**: PyTorch (Preview / Nightly) + CUDA 13.0
* **Torchvision**: 0.21.0+cu124

### 环境配置指南

如需在 RTX 50 系列显卡上充分发挥硬件加速性能，请运行以下命令安装包含最新算力支持的 PyTorch 版本：

```bash
# 安装支持 CUDA 13.0 的预览版
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

**基础依赖安装**：

```bash
pip install pillow numpy
```

## 项目结构

```
ResNet50-Adversarial-Lab/
├── core/
│   ├── loadModel.py          # 基线模型加载与推理（AdversarialModel）
│   └── attack_engine.py      # 对抗攻击引擎（AttackEngine，继承自 AdversarialModel）
├── testset/                  # 测试图像文件夹（放置你想测试的图片）
├── NOTE.md                   # 核心原理笔记（公式、论文、关键概念）
├── EXPERIMENT_LOG.md         # 实验操作日志（时间线记录）
├── report.md                 # 实验报告（阶段成果汇总）
└── README.md                 # 本文件
```

## 快速开始

### 1. 克隆仓库

```bash
git clone <你的仓库地址>
cd ResNet50-Adversarial-Lab
```

### 2. 准备测试图片

将你的测试图片放入 `testset/` 文件夹中。项目已预置几张示例图（cock.jpg, dog.jpg, hen.jpg 等）。

### 3. 运行基线推理（查看模型正常识别结果）

```bash
python core/loadModel.py
```

运行后，你会看到每张测试图片的 Top-1 ~ Top-5 预测结果，例如：

```
“dog.jpg” 【dog】 98.50%
    Top-2: 【cat】 0.80%
    ...
```

### 4. 运行对抗攻击（将图片误导为指定类别）

以下是一个最小可运行的 Python 示例，将 `testset/dog.jpg` 定向攻击为"母鸡"（ImageNet 类别 ID 为 7）：

```python
from PIL import Image
from core.attack_engine import AttackEngine

# 1. 初始化攻击引擎（自动加载 ResNet50 并迁移到 GPU）
engine = AttackEngine()

# 2. 读取并预处理图片
image = Image.open("testset/dog.jpg").convert("RGB")
original_tensor = engine.preprocess(image)

# 3. 查看原始预测
print("原始预测:", engine.predict(original_tensor))

# 4. 生成对抗样本（epsilon 控制扰动强度，越大攻击越强）
adv_tensor, perturbation, data_grad = engine.generate_targeted_adversarial(
    original_tensor, target_id=7, epsilon=0.03
)

# 5. 查看攻击后的预测
print("对抗样本预测:", engine.predict(adv_tensor))

# 6. 将对抗样本还原为可保存的图片
adv_image = engine.tensor_to_image(adv_tensor)
adv_image.save("adversarial_dog.jpg")
print("对抗样本已保存至 adversarial_dog.jpg")
```

**预期效果**：
- 原始图片被模型高置信度识别为 "dog"。
- 对抗样本在人眼看来与原始图片几乎一样，但模型会将其错误识别为 "hen"（母鸡）。
- `epsilon` 是核心调节参数：建议从 `0.01` 开始尝试，逐步增大到 `0.05` 或 `0.1`。

### 5. 关键参数说明

| 参数 | 含义 | 建议取值 |
|------|------|----------|
| `target_id` | 想误导成的目标类别（ImageNet 索引）| 7（母鸡 hen）、... |
| `epsilon` | 扰动强度，每个像素最大变化量 | 0.01 ~ 0.1 |

> **ImageNet 类别查询**：ImageNet-1K 共有 1000 个类别，索引 0~999。常见动物索引示例：7=hen（母鸡）、... 你可以在 `core/loadModel.py` 的 `labels` 中找到完整映射。

## 学习路线（按顺序执行）

本项目按阶段推进，建议按以下顺序阅读代码与运行实验：

1. **阶段一**：阅读 `core/loadModel.py`，理解模型如何加载、预处理、推理。
2. **阶段二**：阅读 `core/attack_engine.py`，理解梯度追踪、定向损失、FGSM 攻击公式。
3. **阶段三（待更新）**：防御手段（对抗训练、图像预处理）。
4. **阶段四（待更新）**：汇总数据，生成最终报告。

## 参考资料

- FGSM 原始论文：[*Explaining and Harnessing Adversarial Examples*](https://arxiv.org/abs/1412.6572), Ian J. Goodfellow et al., ICLR 2015.
