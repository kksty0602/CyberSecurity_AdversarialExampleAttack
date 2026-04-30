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
