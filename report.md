# 实验报告：ResNet50 定向对抗攻击研究

## 1. 实验概述

本实验旨在研究深度学习图像分类模型的安全性，以 PyTorch 官方预训练的 ResNet50 为攻击目标，实现基于梯度的定向对抗样本攻击。实验核心复现了 FGSM（Fast Gradient Sign Method）单步攻击与 PGD（Projected Gradient Descent）迭代攻击两种经典算法，并在本地 NVIDIA RTX 5060 GPU 环境下完成从模型部署、攻击实施到问题诊断与修复的完整流程。

实验过程中发现一个关键实现缺陷：在归一化空间直接执行 `torch.clamp(tensor, 0, 1)` 会严重扭曲扰动方向，导致高置信度目标或大类间攻击失败。通过将攻击下沉至像素空间 `[0, 1]` 并重新设计梯度追踪链，成功修复该问题；进一步引入 PGD 多步迭代攻击，显著提升了语义差距大的定向攻击成功率。

---

## 2. 实验环境

| 组件 | 版本/型号 |
|------|----------|
| 操作系统 | Windows 11 Home China |
| Python | 3.13 |
| 深度学习框架 | PyTorch (Nightly) + CUDA 13.0 |
| Torchvision | 0.21.0+cu124 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU (sm_120) |
| Web 框架 | Streamlit 1.57.0 |
| 目标模型 | ResNet50 (ImageNet-1K 预训练权重) |

---

## 3. 阶段一：基线工程与模型部署

### 3.1 模型加载与预处理

使用 `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)` 加载官方预训练权重，并调用 `weights.transforms()` 获取与当前权重严格匹配的标准预处理管道，包含：

1. `Resize(256)`：将短边缩放至 256 像素；
2. `CenterCrop(224)`：中心裁剪为 224×224；
3. `ToTensor()`：映射到 `[0, 1]` 并转换为张量；
4. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`：ImageNet 标准归一化。

模型自动检测 CUDA 设备并挂载至 `cuda:0`，显式调用 `model.eval()` 进入推理模式，关闭 Dropout 与 BatchNorm 的统计更新，保证结果确定性。

### 3.2 批量推理与 Top-k 输出

遍历本地 `testset` 目录，对常见图像格式（`.jpg`, `.png`, `.bmp` 等）执行批量推理。输出格式包含：

- **Top-1**：置信度最高的类别名称与百分比；
- **Top-2 ~ Top-5**：次高类别的详细信息。

推理过程使用 `torch.inference_mode()` 替代传统的 `torch.no_grad()`，在禁用梯度计算的基础上进一步关闭视图跟踪（view-tracking）与版本计数器（version counters），获得更优的推理速度与显存效率。

### 3.3 面向对象封装与 Web 适配

将模型加载、预处理、推理、设备管理统一封装为 `AdversarialModel` 类，遵循单一职责原则：

- `preprocess(image)`：支持 PIL Image 与 NumPy 数组输入；
- `predict(image_tensor)`：返回结构化的 `Dict`（含 `topk_ids` / `topk_names` / `topk_confs`），便于 Web 端序列化为 JSON；
- `validate_image_path(path)`：输入校验，拦截非法格式或损坏图片，防止后端崩溃。

在文件底部预留 `@st.cache_resource` 工厂函数注释模板，供后续 Streamlit Web 端直接启用单例缓存，避免 200MB+ 权重的重复加载。

---

## 4. 阶段二：对抗攻击实施

### 4.1 攻击引擎基础架构

在 `core/attack_engine.py` 中构建 `AttackEngine` 类，继承 `AdversarialModel`，扩展对抗攻击所需的梯度追踪与清理能力。

**`prepare_adversarial_input`**：通过 `.clone().detach()` 复制原始输入张量，避免破坏原始数据；对克隆后的张量显式调用 `.requires_grad_(True)`，开启输入域梯度追踪。这是 Autograd 记录从输出层到输入层完整计算图的前提，没有此步骤，`loss.backward()` 无法传播到像素层，攻击将失去方向依据。

**`zero_gradients`**：在每次计算新的攻击梯度前，调用 `self.model.zero_grad()` 清空模型参数梯度；若传入输入张量，则同时清空其累积梯度。在迭代式攻击（如 PGD）中，此机制确保每步攻击方向独立，不受前次梯度残留干扰。

### 4.2 定向损失计算与梯度提取

**定向攻击策略**：与非定向攻击"远离"正确答案（最大化真实标签损失）相反，定向攻击的目标是"靠近"指定错误答案。使用 `CrossEntropyLoss(output, target_id)` 衡量模型 logits 与目标类别的差距，**最小化该损失等价于最大化模型对 target_id 的置信度**。

**梯度提取**：调用 `loss.backward()` 后，PyTorch Autograd 自动应用链式法则，从损失节点回溯至输入节点：

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial f_{out}} \cdot \frac{\partial f_{out}}{\partial f_{n-1}} \cdots \frac{\partial f_{1}}{\partial X}$$

得到的 `data_grad` 形状与输入图像相同，每个元素表示对应像素通道的修改灵敏度。FGSM 利用该梯度的符号构造扰动，忽略大小，确保扰动在每个像素上的绝对值相同。

### 4.3 FGSM 单步攻击实现

基于 Goodfellow et al. (ICLR 2015) 的原始公式实现定向 FGSM：

$$X_{adv} = X - \epsilon \cdot \text{sign}(\nabla_X J(X, Y_{target}))$$

实现流程：
1. 准备对抗输入（克隆 + 梯度追踪）；
2. 计算定向损失；
3. 反向传播提取梯度；
4. 构造扰动：`perturbation = epsilon * sign(data_grad)`；
5. 定向攻击使用减号：`adv_tensor = original_tensor - perturbation`；
6. 返回三元组：`adv_tensor`（对抗样本）、`perturbation`（纯噪声，用于热力图可视化）、`data_grad`（原始梯度）。

**初始版本的缺陷**：实现中直接在归一化后的张量上执行 `torch.clamp(adv_tensor, 0.0, 1.0)`。由于 ImageNet 归一化将像素从 `[0, 1]` 映射到约 `[-2.1, 2.2]`，此 `clamp` 会把大量合法像素硬生生截断到边界值，导致扰动方向严重扭曲。

### 4.4 像素空间攻击重构（关键 Bug 修复）

**问题诊断**：在本地命令行测试中发现，对于高置信度（70%+）的 banana 图片，即使 `ε=0.1`，攻击后 banana 置信度仅从 10% 微弱下降，且 `ε=0.1` 时甚至出现置信度从 9.97% 反弹至 21.48% 的反常现象。深入分析后定位根因：

- 归一化空间中的 `clamp(0, 1)` 将负数像素全部压到 0，大于 1 的压到 1；
- 重新归一化后，这些边界像素的实际扰动与预期梯度方向完全无关；
- 大量像素同时被推向边界，破坏了攻击的系统性方向。

**修复方案**：将攻击彻底下沉至像素空间 `[0, 1]`：

1. 反归一化：`pixel_input = original_tensor * std + mean`；
2. 在**像素空间张量**上开启 `requires_grad=True`；
3. 重新归一化后送入模型（乘除法均可微，Autograd 能将梯度顺畅传播回像素空间）；
4. 反向传播得到 `grad_pixel`；
5. 在像素空间执行 FGSM：`pixel_adv = clamp(pixel_input - epsilon * sign(grad_pixel), 0, 1)`；
6. 最后再归一化一次：`adv_tensor = (pixel_adv - mean) / std`。

此修复确保 `clamp` 只截断真正的非法像素值，不会扭曲已经算好的梯度方向。

### 4.5 PGD 迭代攻击实现

FGSM 单步攻击对高置信度或语义差距大的目标（如 banana → cock）成功率有限。引入 PGD（Madry et al., ICLR 2018）迭代攻击，核心思想是"小碎步 + 反复修正"。

**算法公式**：

$$X_{t+1} = \Pi_{\epsilon} \left( \text{clamp}\left( X_t - \alpha \cdot \text{sign}(\nabla_X J(X_t, Y_{target})), 0, 1 \right) \right)$$

**实现细节**：
- 每步步长 `alpha = epsilon / 4`（默认）；
- 每步更新后执行**投影操作** `Π_ε`：计算当前对抗样本与原始样本的偏差，用 `torch.clamp(perturbation, -epsilon, epsilon)` 限制 $L_\infty$ 距离不超过 `epsilon`；
- 再执行 `clamp(0, 1)` 截断到合法像素范围；
- 默认迭代 10~20 次，可根据目标难度动态调整。

**优势**：多步迭代使攻击能沿着局部梯度持续优化，逐步逼近目标类别。对于单步 FGSM 难以跨越的决策边界，PGD 通过反复修正累积足够的扰动动量，成功率显著提升。

### 4.6 实验结果与分析

使用 `test_cli.py` 在本地命令行对 `testset` 中的 banana 图片进行定向攻击测试，对比修复前后及两种算法的差异：

**修复前（归一化空间 clamp）**：
- `ε=0.03`：banana 置信度 70.65% → 10.54%（仍为 Top-1），攻击失败；
- `ε=0.05`：banana 置信度 70.65% → 9.97%（仍为 Top-1），攻击失败；
- `ε=0.1`：banana 置信度 70.65% → 21.48%（置信度反弹，仍为 Top-1），攻击失败。
- 共性：第二预测始终为 lemon 等同类黄色水果，说明扰动方向被扭曲为"同类混淆"而非"跨类误导"。

**修复后（像素空间 FGSM）**：
- 对近邻类别（如 banana → lemon/orange）效果明显改善，`ε=0.03` 即可取得较高成功率；
- 对语义差距大的远类别（如 banana → cock），`ε ≤ 0.05` 时单步攻击仍难以跨越决策边界。

**PGD 迭代攻击**：
- 对近邻类别：`ε=0.03`, `iter=10` 即可稳定成功；
- 对远类别（banana → cock）：`ε=0.05`, `iter=20` 成功率远高于 FGSM，能稳定将模型误导至目标类别；
- 实际最大扰动值与输入 `ε` 基本一致，验证像素空间攻击的正确性。

**结论**：
1. 像素空间攻击是正确实现 FGSM/PGD 的前提，归一化空间的错误 `clamp` 会根本性破坏攻击方向；
2. 单步 FGSM 适用于低置信度或近邻类别的快速攻击；
3. PGD 迭代攻击是大类间、高置信度目标的可靠选择，多步累积的扰动动量能有效跨越深层模型的复杂决策边界。

---

## 5. 阶段三展望：防御与 Web UI

### 5.1 防御手段规划

- **图像预处理防御**：实现高斯模糊、JPEG 压缩等预处理，利用对抗噪声的高频特性，在不影响人眼识别的前提下破坏扰动结构；
- **对抗训练**：在训练过程中注入 FGSM/PGD 样本，提升模型鲁棒性。针对预训练模型，可考虑冻结 backbone，仅对最后一层全连接层进行对抗微调。

### 5.2 Web UI 设计要点

- **算法选择器**：支持 FGSM（快速单步预览）与 PGD（高精度迭代）切换；
- **动态参数面板**：FGSM 模式下仅显示 `ε` 滑块；PGD 模式下额外显示迭代次数滑块（5~50）与实时迭代进度；
- **可视化对比**：并排展示原始图、噪声热力图、对抗样本图；PGD 模式下增加"目标类别置信度迭代曲线"折线图；
- **一键加固**：集成高斯模糊/JPEG 压缩等防御按钮，实时对比防御前后识别率变化。

---

## 6. 参考文献

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples*. ICLR 2015. https://arxiv.org/abs/1412.6572
2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR 2018. https://arxiv.org/abs/1706.06083
