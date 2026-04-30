# model/loadModel.py
# 第一阶段优化：重构基线代码，完成推理逻辑的面向对象封装，适配 Web 端调用接口

import os
import sys
import json
import urllib.request
import platform
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class AdversarialModel:
    """
    ResNet50 对抗攻击基线模型。

    封装模型加载、ImageNet 标准预处理、Top-k 推理及设备管理，
    设计为可被 Web 后端（Streamlit/Gradio）直接调用的模块化组件。
    """

    IMAGENET_LABELS_URL = (
        "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/"
        "master/imagenet-simple-labels.json"
    )
    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

    def __init__(self, device: Optional[torch.device] = None):
        """
        初始化模型，加载预训练权重并自动检测 CUDA 环境。

        Args:
            device: 指定计算设备，None 则自动检测 CUDA。
        """
        self.device = self.to_device(device)
        print(f"[INFO] AdversarialModel 使用设备: {self.device}")

        # 加载官方预训练权重与配套预处理管道
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.to(self.device)
        # 进入推理模式，关闭 Dropout 与 BatchNorm 的训练时统计更新
        self.model.eval()
        print("[INFO] ResNet50 模型加载完成，已切换至 eval 模式。")

        # 动态获取 ImageNet-1K 人类可读标签
        self.labels = self._load_labels()
        print(f"[INFO] 标签映射加载完成，共 {len(self.labels)} 个类别。")

    def to_device(self, device: Optional[torch.device] = None) -> torch.device:
        """
        显存管理：确保模型与张量严格在目标设备上执行。

        Args:
            device: 指定设备，None 则优先选择 cuda:0。

        Returns:
            torch.device: 实际使用的计算设备。
        """
        if device is not None:
            return device
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _load_labels(self) -> List[str]:
        """从网络加载 ImageNet 标签，失败时回退为数字 ID 列表。"""
        try:
            with urllib.request.urlopen(self.IMAGENET_LABELS_URL, timeout=10) as response:
                labels = json.loads(response.read().decode("utf-8"))
            return labels
        except Exception:
            return [str(i) for i in range(1000)]

    def get_label(self, class_id: int) -> str:
        """根据类别索引返回可读标签，越界时回退为 ID 字符串。"""
        if 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return str(class_id)

    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        封装标准的 ImageNet 预处理逻辑（Resize, CenterCrop, Normalize）。

        Args:
            image: PIL Image 或 NumPy 数组 (H, W, C)。

        Returns:
            torch.Tensor: 预处理后的张量，形状为 (1, C, H, W)。
        """
        # 统一转换为 PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        # 确保为 RGB 模式（避免 PNG 透明度通道干扰预处理）
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 调用与当前权重严格匹配的官方预处理管道
        transforms = self.weights.transforms()
        tensor = transforms(image)  # (C, H, W)
        return tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)

    def predict(self, image_tensor: torch.Tensor, top_k: int = 5) -> Dict[str, Any]:
        """
        输入预处理后的张量，输出结构化的 Top-k 预测结果。

        Args:
            image_tensor: 预处理后的图像张量，形状 (N, C, H, W) 或 (C, H, W)。
            top_k: 返回前 k 个最高概率类别。

        Returns:
            Dict: 包含 'topk_ids'、'topk_names'、'topk_confs' 的结构化字典，
                  便于 Web 端直接序列化为 JSON 响应。
        """
        # 若传入三维张量，自动补全 Batch 维度
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # 确保张量位于正确设备
        image_tensor = image_tensor.to(self.device)

        # 使用 inference_mode 替代 no_grad，获得更快的推理速度并节省显存
        with torch.inference_mode():
            outputs = self.model(image_tensor)  # (N, 1000) logits
            probabilities = F.softmax(outputs, dim=1)  # 将 logits 转换为概率分布
            probs, class_ids = torch.topk(probabilities, k=top_k, dim=1)

        # 移回 CPU 并转为 Python 原生类型，避免跨设备传输开销影响 Web 响应
        probs = probs.squeeze(0).cpu().tolist()
        class_ids = class_ids.squeeze(0).cpu().tolist()

        # 组装结构化输出，不直接 print，便于 Web 端消费
        result = {
            "topk_ids": class_ids,
            "topk_names": [self.get_label(cid) for cid in class_ids],
            "topk_confs": [round(p * 100, 2) for p in probs],
        }
        return result

    def validate_image_path(self, image_path: str) -> bool:
        """
        异常处理：校验输入文件路径与格式合法性。

        Args:
            image_path: 待校验的图像文件路径。

        Returns:
            bool: 是否通过校验。
        """
        if not os.path.isfile(image_path):
            return False
        _, ext = os.path.splitext(image_path)
        if ext.lower() not in self.VALID_EXTENSIONS:
            return False
        return True

    def infer_from_path(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """
        从文件路径加载图像并执行完整推理流程（校验 -> 读取 -> 预处理 -> 预测）。

        Args:
            image_path: 图像文件路径。
            top_k: 返回前 k 个预测结果。

        Returns:
            Dict: 结构化预测结果；校验失败或读取异常时返回包含 error 字段的字典。
        """
        if not self.validate_image_path(image_path):
            return {"error": f"非法图像路径或格式: {image_path}"}

        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.preprocess(image)
            return self.predict(tensor, top_k=top_k)
        except Exception as e:
            return {"error": str(e)}


# ==================== Web UI 缓存适配预留 ====================
#
# 说明：以下工厂函数供 Streamlit Web 端调用。
# 在 Streamlit 环境中，取消下方注释并将装饰器替换为 @st.cache_resource，
# 即可实现模型单例加载，避免网页刷新时重复加载 200MB+ 的权重文件。
#
# try:
#     import streamlit as st
#     _cache_decorator = st.cache_resource
# except ImportError:
#     def _cache_decorator(func=None, **kwargs):
#         if func is not None:
#             return func
#         return lambda f: f
#
# @_cache_decorator
# def get_adversarial_model() -> AdversarialModel:
#     """全局缓存的模型工厂函数。"""
#     return AdversarialModel()


def verify_environment() -> None:
    """
    验证运行环境是否满足项目第一阶段要求：
    Python 3.13、PyTorch Nightly、CUDA 13.0、sm_120 计算能力。
    """
    print("[INFO] 正在验证运行环境...")

    py_major, py_minor, _ = platform.python_version_tuple()
    py_ver = (int(py_major), int(py_minor))
    if py_ver < (3, 13):
        print(f"[WARN] 当前 Python 版本为 {py_major}.{py_minor}，建议升级至 3.13 以获得最佳兼容性。")
    else:
        print(f"[INFO] Python 版本: {py_major}.{py_minor} ✔")

    print(f"[INFO] PyTorch 版本: {torch.__version__}")
    if torch.cuda.is_available():
        cuda_ver = torch.version.cuda
        print(f"[INFO] CUDA 版本: {cuda_ver}")
        if cuda_ver and not cuda_ver.startswith("13"):
            print(f"[WARN] 当前 CUDA 版本为 {cuda_ver}，建议安装支持 CUDA 13.0 的 PyTorch Nightly 以完全释放 RTX 5060 算力。")
        else:
            print("[INFO] CUDA 13.0 检查通过 ✔")

        capability = torch.cuda.get_device_capability()
        print(f"[INFO] GPU 计算能力: sm_{capability[0]}{capability[1]}")
        if capability != (12, 0):
            print(f"[WARN] 当前 GPU 计算能力为 sm_{capability[0]}{capability[1]}，预期为 sm_120 (RTX 5060)。")
        else:
            print("[INFO] sm_120 计算能力检查通过 ✔")

        print(f"[INFO] GPU 型号: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA 不可用，将回退至 CPU 推理，性能可能受限。")


def main():
    """批量推理入口：验证环境、加载模型、遍历 testset 并输出 Top-1 ~ Top-5 结果。"""
    verify_environment()
    print()

    # 实例化模型（Web 端建议通过缓存工厂函数获取单例）
    model = AdversarialModel()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    testset_dir = os.path.join(os.path.dirname(script_dir), "testset")

    if not os.path.isdir(testset_dir):
        print(f"[ERROR] 测试目录不存在: {testset_dir}")
        sys.exit(1)

    image_files = [
        f for f in os.listdir(testset_dir)
        if f.lower().endswith(AdversarialModel.VALID_EXTENSIONS)
    ]
    image_files.sort()

    if not image_files:
        print(f"[WARN] 在 {testset_dir} 中未找到支持的图像文件。")
        sys.exit(0)

    success_count = 0

    print("=" * 60)
    print("开始批量推理...")
    print("=" * 60 + "\n")

    for filename in image_files:
        file_path = os.path.join(testset_dir, filename)

        # 调用封装后的推理接口，返回结构化字典而非直接 print
        result = model.infer_from_path(file_path, top_k=5)

        # 异常处理：非法图片上传时不会导致后端服务崩溃
        if "error" in result:
            print(f"[SKIP] {filename}: {result['error']}")
            continue

        top1_name = result["topk_names"][0]
        top1_conf = result["topk_confs"][0]

        print(f"“{filename}” 【{top1_name}】 {top1_conf:.2f}%")

        for rank in range(1, 5):
            name = result["topk_names"][rank]
            conf = result["topk_confs"][rank]
            print(f"    Top-{rank + 1}: 【{name}】 {conf:.2f}%")

        success_count += 1
        print()

    print("=" * 60)
    print(f"推理完成。成功识别图片总数: {success_count} / {len(image_files)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
