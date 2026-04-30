# 实验报告：ResNet50 定向对抗攻击研究

## 阶段二：对抗攻击实施

### 任务 2.1 — 攻击引擎基础架构

**目标**：搭建可复用的对抗攻击基座，支持梯度追踪与梯度清理。

**实现内容**：
- 在 `core/attack_engine.py` 中构建 `AttackEngine` 类，继承 `AdversarialModel`。
- `prepare_adversarial_input`：通过 `.clone().detach()` 复制原始输入，显式开启 `requires_grad=True`，确保 Autograd 能记录从输出到输入的完整计算图。
- `zero_gradients`：调用 `self.model.zero_grad()` 与 `input_tensor.grad.zero_()`，防止前次攻击残留梯度干扰本次方向。

**关键结论**：梯度追踪是对抗攻击的前提；没有 `requires_grad=True`，反向传播无法到达像素层，攻击算法将失去方向依据。

---

### 任务 2.2 — 定向损失计算与梯度提取

**目标**：让模型"认错"到指定类别，并提取像素级修改建议。

**实现内容**：
- `compute_targeted_loss(input_tensor, target_id)`：使用 `CrossEntropyLoss` 衡量当前 logits 与目标类别的差距。**最小化该损失等价于最大化模型对 target_id 的置信度**。
- `extract_gradient(loss, input_tensor)`：调用 `loss.backward()`，通过链式法则将误差从输出层逐层传回输入层，得到 `∂Loss/∂X`（Data Gradient）。

**关键结论**：
- 定向攻击与非定向攻击策略相反：非定向攻击"远离"正确答案（最大化真实标签损失），定向攻击"靠近"指定错误答案（最小化目标标签损失）。
- `loss.backward()` 自动应用链式法则，无需手动推导各层导数。

---

### 任务 2.3 — FGSM 对抗样本生成与参数接口封装

**目标**：将数学公式转化为可运行的代码，生成肉眼难辨但模型会认错的对抗图像。

**实现内容**：
- `generate_targeted_adversarial(original_tensor, target_id, epsilon)`：
  1. 准备对抗输入（克隆 + 梯度追踪）；
  2. 计算定向损失；
  3. 反向传播提取梯度；
  4. 按 FGSM 公式构造扰动：`perturbation = epsilon * sign(data_grad)`；
  5. 定向攻击使用减号：`adv_tensor = original_tensor - perturbation`；
  6. 使用 `torch.clamp` 将像素截断到 [0, 1]，防止数值溢出。
- `tensor_to_image(tensor)`：执行 ImageNet 反归一化（x = x_norm * std + mean），将张量还原为 PIL Image，便于可视化与保存。

**关键结论**：
- `epsilon` 是攻击的"油门"：值越大，扰动越强，模型越容易被骗，但图像失真也越明显；值越小，对抗样本越隐蔽，但攻击成功率可能下降。
- `torch.clamp` 是图像质量的保险栓：不加截断，像素值可能溢出 [0, 1]，反归一化后产生肉眼可见的色斑或灰块。
- 返回 `perturbation` 独立噪声张量，使后续"噪声热力图"可视化成为可能。

---

## 下一阶段展望

阶段三将聚焦防御与加固：通过对抗训练、图像预处理（如高斯模糊、JPEG 压缩）等手段，提升模型对对抗样本的鲁棒性，并量化防御前后的识别率变化。
