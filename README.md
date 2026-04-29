# ResNet50-Adversarial-Lab

基于 PyTorch 的 ResNet50 图像识别与目标定向对抗攻击（Targeted Adversarial Attack）研究项目。

## 📖 项目简介

本项目旨在探索和复现深度卷积神经网络（CNN）在计算机视觉任务中的表现及其安全性。项目以官方预训练的 ResNet50 模型为基础，包含两个核心实验模块：
1. **基线推理（Baseline Inference）**：实现对本地图像数据集的高效批量分类与置信度评估。
2. **对抗样本攻击（Adversarial Attack）**：利用梯度下降算法（如 FGSM/PGD）生成微小扰动，实现目标定向攻击（例如：将模型识别高度自信的“犬类”图像，通过像素级干扰，定向欺骗为“母鸡”）。

## ⚙️ 硬件与运行环境

本项目已在较新的硬件架构上完成测试与适配。针对 NVIDIA RTX 50 系列显卡（sm_120 架构），标准稳定版 PyTorch 可能会出现算力不兼容警告，因此推荐使用 Nightly 预览版。

* **操作系统**: Windows
* **计算设备**: NVIDIA GeForce RTX 5060 Laptop GPU
* **编程语言**: Python 3.13
* **深度学习框架**: PyTorch (Preview / Nightly) + CUDA 13.0

### 环境配置指南

如需在 RTX 50 系列显卡上充分发挥硬件加速性能，请运行以下命令安装包含最新算力支持的 PyTorch 版本：

```bash

# 安装支持 CUDA 13.0 的预览版
pip3 install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu130](https://download.pytorch.org/whl/nightly/cu130)