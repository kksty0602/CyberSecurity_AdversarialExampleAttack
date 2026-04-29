# 核心原理笔记

## ResNet50 推理前向传播

- **模型输出**：ResNet50 最后一层为全连接层，输出 1000 维 logits（对应 ImageNet-1K 的 1000 个类别）。
- **Softmax**：将 logits 转换为概率分布。
  $$P(y_i \mid \mathbf{x}) = \frac{e^{z_i}}{\sum_{j=1}^{1000} e^{z_j}}$$
- **Top-k**：利用 `torch.topk` 提取概率最高的 k 个类别索引。
- **推理模式**：`model.eval()` 关闭 Dropout 与 BatchNorm 的统计更新，保证结果确定性。
