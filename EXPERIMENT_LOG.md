# 实验日志

## 2026/04/29 - 阶段一：模型部署脚本编写

**修改内容**：
- 新增 `model/resnet50_deploy.py`，实现基于 PyTorch 预训练 ResNet50 的模型部署类。
- 采用面向对象设计（`ResNet50Deploy`），封装图像预处理、模型推理与批量评估逻辑。
- 支持自动检测 CUDA 设备，充分利用 RTX 5060 加速。
- 输出格式包含图像名称、Top-1 预测标签及置信度、Top-5 预测列表。
- 自动下载 ImageNet 类别索引文件，用于将模型输出映射为人类可读标签。

**实验预期**：
- 运行脚本后，能够正确遍历 `testset` 文件夹中的所有图像并输出分类结果。
- 若 GPU 可用，推理过程应在 CUDA 上完成，显著提速。

## 2026/04/29 - 任务 1.2：构建批量推理基准

**修改内容**：
- 新增 `model/loadModel.py`，严格按照项目规范实现 ResNet50 批量推理脚本。
- 使用 `ResNet50_Weights.DEFAULT` 加载官方权重，并调用 `weights.transforms()` 获取标准预处理管道。
- 设备自动检测并挂载 CUDA，模型显式调用 `eval()` 进入推理模式。
- 动态请求 ImageNet-1K 标签 JSON，失败时回退至 Class ID。
- 遍历 `testset` 目录，对常见图像格式进行批量推理，输出 Top-1 ~ Top-5 结果。
- 增加鲁棒性处理：`testset` 缺失异常、单张图片损坏 `try-except` 跳过。

**实验预期**：
- 脚本可直接运行，正确输出每张图像的 Top-1 识别标签与置信度（保留两位小数），并附 Top-2 ~ Top-5 详情。
- 若网络可达，标签为人类可读英文单词；否则显示数字 ID。

## 2026/04/30 - 任务 1.2：重构模型加载脚本（面向对象 + 环境验证 + 全局缓存）

**修改内容**：
- 重构 `model/loadModel.py`，将过程式代码升级为面向对象的 `ResNet50Predictor` 类。
- 新增 `verify_environment()` 函数，自动检查 Python 版本、PyTorch 版本、CUDA 版本及 GPU 计算能力（sm_120）。
- 引入 `@_st_cache_resource` 兼容层，在 Streamlit 环境下可自动启用 `st.cache_resource` 全局缓存，避免 Web 端重复加载模型。
- 保留批量推理基准测试功能，输出格式维持 Top-1 主结果 + Top-2 ~ Top-5 缩进详情。
- 关键步骤补充中文注释，符合项目技术约束。

**实验预期**：
- 脚本运行时首先打印环境验证报告，确保 RTX 5060 的 CUDA 13.0 与 sm_120 算力得到正确识别。
- `ResNet50Predictor` 类可被 Web 模块直接导入复用，模型实例在 Streamlit 会话中仅加载一次。
- 批量推理结果与重构前保持一致，Top-1/Top-5 置信度输出无误。

## 2026/04/30 - 优化：第一阶段代码重构与 Web 适配

**修改内容**：
- 重构 `model/loadModel.py`，取消零散推理脚本，统一封装为 `AdversarialModel` 类。
- `__init__` 负责加载 `ResNet50_Weights.DEFAULT` 权重并自动检测 CUDA 环境（RTX 5060）。
- `preprocess(image)` 封装标准 ImageNet 预处理（Resize, CenterCrop, Normalize），支持 PIL Image 与 NumPy 数组输入。
- `predict(image_tensor)` 返回结构化的 Dict（含 topk_ids / topk_names / topk_confs），不再直接 print，便于 Web 端序列化为 JSON。
- 新增 `to_device()` 方法显式管理显存，确保推理严格在 `cuda:0` 执行；使用 `torch.inference_mode()` 替代 `torch.no_grad()` 以提升推理速度。
- 增加 `validate_image_path()` 输入校验，非法格式或损坏图片不会导致后端崩溃。
- 在文件底部预留 `get_adversarial_model()` 工厂函数与 `@st.cache_resource` 注释模板，供后续 Streamlit Web 端直接启用单例缓存。

**实验预期**：
- `AdversarialModel` 可被 Web 模块直接导入复用，数据流完全解耦。
- 预留的缓存接口在接入 Streamlit 后只需取消注释即可生效，避免重复加载 200MB+ 权重。
- 批量推理输出与重构前保持一致，异常输入可被安全拦截。

## 2026/04/30 - 阶段二任务 2.1：构建定向攻击引擎基础架构

**修改内容**：
- 新建 `core/attack_engine.py`，继承 `AdversarialModel` 构建 `AttackEngine` 攻击引擎类。
- 添加 `prepare_adversarial_input(image_tensor)` 方法：通过 `.clone().detach()` 复制原始张量，显式调用 `.requires_grad_(True)` 开启输入域梯度追踪，并确保张量与模型均处于 `self.device`（RTX 5060 CUDA）上。
- 添加 `zero_gradients(input_tensor)` 辅助方法：在每次计算新的攻击梯度前，调用 `self.model.zero_grad()` 清空模型参数梯度；若传入输入张量，则同时清空其累积梯度，防止梯度累积导致攻击方向偏差。
- 在 `prepare_adversarial_input` 中自动调用 `self.model.eval()`，确保攻击阶段模型处于确定性推理态。
- 关键步骤补充中文注释，解释梯度追踪在对抗攻击中的必要性（Autograd 计算 Data Gradient 的前提）。

**实验预期**：
- `AttackEngine` 可被后续 FGSM/PGD 实现直接继承或调用，提供标准化的对抗输入准备与梯度清理能力。
- 在 RTX 5060 上，攻击张量与模型参数均位于 `cuda:0`，推理与反向传播均在 GPU 上完成。
- 连续生成多个对抗样本时，梯度清零机制确保每次攻击方向独立，不受前次梯度残留干扰。

## 2026/04/30 - 阶段二任务 2.2：实现定向损失计算与梯度提取

**修改内容**：
- 在 `core/attack_engine.py` 的 `AttackEngine` 类中新增 `compute_targeted_loss(input_tensor, target_id)` 方法。
- 使用 `torch.nn.CrossEntropyLoss()` 计算模型输出 logits 与指定目标类别 `target_id`（如 7 代表母鸡）之间的定向损失。
- 方法内部显式调用 `self.model.eval()`，关闭 Dropout 等随机层，确保攻击过程中梯度计算稳定、可复现。
- 新增 `extract_gradient(loss, input_tensor)` 方法：对损失调用 `.backward()`，通过链式法则反向传播至输入层；从 `input_tensor.grad.data` 中提取数据域梯度 `data_grad`。
- 在 `extract_gradient` 开头显式调用 `self.model.zero_grad()` 与 `input_tensor.grad.zero_()`，彻底清理残余梯度，防止历史推理干扰本次攻击。
- 补充中文注释，解释定向攻击中为何针对 `target_id` 计算损失（最小化目标类别损失等价于最大化模型对目标类别的置信度），以及 `loss.backward()` 如何通过链式法则将分类误差转化为像素修改建议。

**实验预期**：
- 给定一张测试图与目标类别 ID，`compute_targeted_loss` 能返回可反向传播的损失标量。
- `extract_gradient` 能正确输出形状与输入图像相同的梯度张量，且梯度非零，表明链式法则已成功将误差从输出层传回像素层。
- 整个求导过程在 RTX 5060 的 CUDA 环境下完成，无显存泄漏或残余梯度干扰。

## 2026/04/30 - 阶段二任务 2.3：生成对抗扰动并封装参数接口

**修改内容**：
- 在 `core/attack_engine.py` 的 `AttackEngine` 类中新增 `generate_targeted_adversarial(original_tensor, target_id, epsilon)` 方法。
- 严格按照 FGSM 定向攻击公式实现：`X_adv = X - epsilon * sign(∇_X J(X, Y_target))`。
- 方法内部自动串联 `prepare_adversarial_input` → `compute_targeted_loss` → `extract_gradient` → 构造扰动 → 截断像素值。
- 使用 `torch.clamp(adv_tensor, 0.0, 1.0)` 将对抗样本像素限制在合法范围，防止反归一化后图像溢出失真。
- 返回三元组：`adv_tensor`（对抗样本）、`perturbation`（纯噪声，用于热力图可视化）、`data_grad`（原始梯度）。
- 新增 `tensor_to_image(tensor)` 方法：执行 ImageNet 反归一化（x = x_norm * std + mean），将张量还原为 PIL Image，便于后续可视化与保存。
- `tensor_to_image` 中再次调用 `torch.clamp` 确保输出像素在 [0, 1]，映射到 [0, 255] 后生成 uint8 图像。

**实验预期**：
- 给定一张测试图、目标类别（如 7 代表母鸡）与 epsilon（如 0.03），`generate_targeted_adversarial` 能返回可正确显示的对抗样本。
- 原始图像与对抗样本在人眼上几乎无法区分（epsilon 较小时），但模型预测结果从原标签变为 target_id。
- `tensor_to_image` 可直接将预处理后的张量或对抗样本转换为 PIL Image，用于 Matplotlib/Streamlit 展示。
- 整个流程在 RTX 5060 的 CUDA 环境下完成，无显存泄漏。

---
