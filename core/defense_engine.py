"""
core/defense_engine.py
阶段四：防御引擎，实现图像预处理防御手段。
"""

import io
from typing import Literal

from PIL import Image, ImageFilter


class DefenseEngine:
    """
    预处理防御引擎。

    提供基于图像预处理的防御方法，通过破坏对抗扰动的结构化特征
    来降低对抗样本的攻击效果。
    """

    def gaussian_defense(self, image: Image.Image, sigma: float) -> Image.Image:
        """
        高斯模糊防御。

        原理：对抗扰动通常具有高频、局部化的特征。高斯模糊通过低通滤波
        平滑图像中的高频噪声，破坏对抗扰动的空间结构，使模型恢复正确识别。

        Args:
            image: 输入图像（PIL RGB）。
            sigma: 高斯核标准差，越大模糊程度越高（0.5 ~ 5.0）。

        Returns:
            Image.Image: 模糊后的图像。
        """
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    def jpeg_defense(self, image: Image.Image, quality: int) -> Image.Image:
        """
        JPEG 压缩防御。

        原理：JPEG 是有损压缩，在量化过程中会丢弃高频信息。
        对抗扰动中的许多精细结构属于高频成分，经 JPEG 压缩后会被抑制或消除。

        Args:
            image: 输入图像（PIL RGB）。
            quality: JPEG 质量因子（10 ~ 100），越低压缩越激进、防御效果越强。

        Returns:
            Image.Image: 经 JPEG 压缩后再解码的图像。
        """
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def apply_defense(
        self,
        image: Image.Image,
        method: Literal["gaussian", "jpeg"],
        **kwargs,
    ) -> Image.Image:
        """
        统一防御接口，根据方法名自动分发。

        Args:
            image: 输入图像。
            method: "gaussian" 或 "jpeg"。
            **kwargs: 对应防御方法的参数（sigma 或 quality）。

        Returns:
            Image.Image: 防御处理后的图像。
        """
        if method == "gaussian":
            return self.gaussian_defense(image, kwargs.get("sigma", 1.0))
        elif method == "jpeg":
            return self.jpeg_defense(image, kwargs.get("quality", 75))
        else:
            raise ValueError(f"不支持的防御方法: {method}")
