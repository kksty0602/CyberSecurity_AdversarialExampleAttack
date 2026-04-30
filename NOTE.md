# 核心原理笔记

## ResNet50 推理前向传播

- **模型输出**：ResNet50 最后一层为全连接层，输出 1000 维 logits（对应 ImageNet-1K 的 1000 个类别）。
- **Softmax**：将 logits 转换为概率分布。
  $$P(y_i \mid \mathbf{x}) = \frac{e^{z_i}}{\sum_{j=1}^{1000} e^{z_j}}$$
- **Top-k**：利用 `torch.topk` 提取概率最高的 k 个类别索引。
- **推理模式**：`model.eval()` 关闭 Dropout 与 BatchNorm 的统计更新，保证结果确定性。

## `torch.inference_mode()` vs `torch.no_grad()`

- **`torch.no_grad()`**：禁用梯度计算，减少显存占用并加速推理，但保留部分自动求导元数据。
- **`torch.inference_mode()`**（PyTorch 1.9+）：在 `no_grad` 基础上进一步禁用视图跟踪（view-tracking）与版本计数器（version counters），推理速度更快，且明确声明“此处不会进行反向传播”。
- **应用**：在纯推理路径（如 Web 端批量预测）中优先使用 `inference_mode`，可获得更优的性能表现。

## 面向对象封装与 Web 适配

- **单一职责**：将模型加载、预处理、推理、设备管理拆分为独立方法，便于单元测试与后续维护。
- **数据流解耦**：`predict()` 返回结构化 Dict 而非直接 print，使 Web 框架（Streamlit/Gradio/FastAPI）可直接将结果序列化为 JSON 响应。
- **缓存预留**：通过注释模板预留 `@st.cache_resource` 工厂函数，确保在 Web 会话生命周期内模型仅实例化一次，避免 200MB+ 权重的重复加载与显存泄漏。

## 对抗攻击中的梯度追踪原理

- **Data Gradient**：在基于梯度的对抗攻击（FGSM/PGD）中，核心需求是计算损失函数对输入图像的梯度 $\nabla_X J(X, Y)$。只有将输入张量的 `requires_grad` 设为 `True`，PyTorch Autograd 才能在反向传播时记录从输出到输入的完整计算图，从而得到 $\partial \text{Loss} / \partial X$。
- **FGSM 原始公式**（Goodfellow et al., ICLR 2015）：
  - 非定向攻击：$X_{adv} = X + \epsilon \cdot \text{sign}(\nabla_X J(X, Y_{true}))$
  - 定向攻击：$X_{adv} = X - \epsilon \cdot \text{sign}(\nabla_X J(X, Y_{target}))$
- **论文地址**：[*Explaining and Harnessing Adversarial Examples*](https://arxiv.org/abs/1412.6572), Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy, ICLR 2015.
- **梯度清零必要性**：在迭代攻击或多轮样本生成中，若不清空梯度，前一轮的 `grad` 会累积到本轮，导致扰动方向偏离真实的损失上升/下降方向，攻击成功率显著下降。

## 定向损失与反向传播链式法则

- **CrossEntropyLoss 在定向攻击中的作用**：
  定向攻击的目标是迫使模型将输入错分为指定类别 $Y_{target}$。
  `CrossEntropyLoss(output, target_id)` 衡量模型输出 logits 与目标类别的差距；**损失越小**，模型对目标类别的 Softmax 概率越高。
  因此，通过降低该损失，即可实现"让模型认错"的效果。
- **链式法则（Chain Rule）**：
  `loss.backward()` 自动执行反向传播：
  $$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial f_{out}} \cdot \frac{\partial f_{out}}{\partial f_{n-1}} \cdots \frac{\partial f_{1}}{\partial X}$$
  其中 $\partial L / \partial X$ 即为 Data Gradient，其每个元素对应一个像素通道的修改灵敏度。
  FGSM 利用该梯度的符号构造扰动：$X_{adv} = X - \epsilon \cdot \text{sign}(\nabla_X J(X, Y_{target}))$。

## FGSM 对抗样本生成与像素截断

- **FGSM 定向攻击公式**（Goodfellow et al., ICLR 2015）：
  $$X_{adv} = X - \epsilon \cdot \text{sign}(\nabla_X J(X, Y_{target}))$$
  - 非定向攻击用加号（最大化真实标签损失）；**定向攻击用减号**（最小化目标标签损失）。
  - $\epsilon$ 控制扰动强度，值越大攻击越明显、成功率越高，但人眼可察觉。
  - `sign()` 只保留梯度方向（+1 / -1 / 0），忽略大小，确保扰动在每个像素上绝对值相同。
- **像素截断（Clamp）的必要性**：
  归一化后的图像像素值原本在 [0, 1] 之间。加上或减去扰动后可能溢出该范围（如 < 0 或 > 1）。
  使用 `torch.clamp(tensor, 0.0, 1.0)` 将像素硬性截断回合法区间，保证反归一化后不会生成无效的灰度/色彩值。
- **反归一化（Denormalize）**：
  预处理时执行了 `Normalize(mean, std)`，因此还原图像需执行逆运算：
  $$x = x_{norm} \times \text{std} + \text{mean}$$
  其中 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]。
