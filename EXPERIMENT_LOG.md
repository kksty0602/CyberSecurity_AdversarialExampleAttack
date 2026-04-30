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

---

## 2026/04/30 - 优化：FGSM 攻击下沉至像素空间与 PGD 迭代攻击实现

**修改内容**：

- 重构 `core/attack_engine.py` 中的 `generate_targeted_adversarial` 方法：将攻击空间从归一化空间迁移到像素空间 `[0, 1]`。
- 核心改动：在像素空间开启 `requires_grad=True`，反向传播得到 `grad_pixel`，直接在像素空间执行 `clamp` 后再重新归一化，彻底避免归一化空间 `torch.clamp(tensor, 0, 1)` 对边界像素扰动方向的扭曲。
- 新增 `generate_targeted_pgd` 方法：实现 PGD（Projected Gradient Descent）定向迭代攻击，支持多步小碎步（步长 `alpha = epsilon / 4`）逼近目标，每步投影回 `epsilon` 邻域并截断到合法像素范围。
- `test_cli.py` 同步更新：支持选择 FGSM（单步）或 PGD（迭代）攻击方式，并显示实际最大扰动值以验证像素空间攻击的正确性。

**实验预期**：

- 像素空间 FGSM 对近邻类别（如 banana → lemon）应能在 `ε=0.03` 内取得较高成功率，验证代码逻辑正确。
- PGD 迭代攻击（如 `ε=0.05`, `iter=20`）对高置信度、语义差距大的大类间目标（如 banana → cock）成功率应显著高于单步 FGSM。
- 命令行输出中的"实际最大扰动"应与输入 `ε` 基本一致，确认梯度传播与扰动生成未受归一化/反归一化流程扭曲。

---
