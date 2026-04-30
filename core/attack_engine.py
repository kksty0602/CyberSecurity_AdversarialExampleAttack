# core/attack_engine.py
# 阶段二任务 2.1：构建定向攻击引擎基础架构
# 功能：基于 AdversarialModel 扩展攻击相关基础方法，为后续 FGSM/PGD 定向攻击提供支持

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from core.loadModel import AdversarialModel


class AttackEngine(AdversarialModel):
    """
    定向对抗攻击引擎。

    继承 AdversarialModel 的全部推理与预处理能力，
    扩展对抗样本生成所需的基础方法（梯度追踪、梯度清理、攻击环境预设）。
    """

    def prepare_adversarial_input(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        准备可用于对抗攻击的输入张量。

        核心步骤：
        1. 将原始张量通过 .clone().detach() 复制，确保不破坏原始输入数据；
        2. 对克隆后的张量显式调用 .requires_grad_(True)，开启梯度追踪。

        为什么必须开启梯度追踪：
        在定向 FGSM 攻击中，我们需要计算损失函数对输入图像的梯度（即 Data Gradient）。
        只有将输入张量的 requires_grad 设为 True，PyTorch 的 Autograd 引擎才会在反向传播时
        记录并计算 ∂Loss/∂X，从而指导扰动方向的生成。若关闭梯度追踪，则无法获取输入域的梯度，
        攻击算法将失去方向依据，导致对抗样本生成失败。

        Args:
            image_tensor: 经过预处理的图像张量，形状通常为 (1, C, H, W) 或 (C, H, W)。

        Returns:
            torch.Tensor: 已开启梯度追踪的克隆张量，位于 self.device 上。
        """
        # 攻击环境预设：确保模型处于评估模式，关闭 Dropout 与 BatchNorm 的统计更新，
        # 保证前向传播与梯度计算的确定性，避免训练态引入随机性导致攻击方向不稳定。
        self.model.eval()

        # 硬件对齐：将张量迁移到目标设备（RTX 5060 CUDA），并克隆分离以避免修改原始数据
        adv_input = image_tensor.clone().detach().to(self.device)

        # 开启输入张量的梯度追踪，这是计算对抗扰动方向的前提
        adv_input.requires_grad_(True)

        return adv_input

    def zero_gradients(self, input_tensor: Optional[torch.Tensor] = None) -> None:
        """
        清空模型参数及输入张量的累积梯度，防止梯度累积导致攻击方向偏差。

        在迭代式攻击（如 PGD）或连续生成多个对抗样本时，
        若不及时清零梯度，前一次计算的梯度会累加到本次，导致扰动方向偏离真实的损失上升方向，
        最终降低攻击成功率或产生不可预测的对抗样本。

        Args:
            input_tensor: 可选，外部持有的待清零输入张量（如 prepare_adversarial_input 的返回值）。
                          若提供且存在 grad 属性，则将其梯度清零。
        """
        # 清空模型中所有可训练参数的梯度
        self.model.zero_grad()

        # 若外部传入输入张量，则同时清空其累积梯度
        if input_tensor is not None and input_tensor.grad is not None:
            input_tensor.grad.zero_()

    def compute_targeted_loss(self, input_tensor: torch.Tensor, target_id: int) -> torch.Tensor:
        """
        计算定向攻击损失：迫使模型将输入错误分类为指定的 target_id。

        核心逻辑：
        使用 CrossEntropyLoss 衡量模型当前输出 logits 与目标类别 target_id 之间的差距。
        在定向攻击中，我们的目标不是让模型"远离"正确答案（非定向），
        而是让模型"靠近"我们指定的错误答案（如将"狗"识别为"鸡"）。
        最小化 CrossEntropyLoss(output, target_id) 等价于最大化模型对 target_id 的置信度。

        Args:
            input_tensor: 已开启梯度追踪的对抗输入张量，形状 (1, C, H, W)。
            target_id: 目标类别在 ImageNet-1K 中的索引（如 7 代表母鸡 hen）。

        Returns:
            torch.Tensor: 标量损失值，位于计算图中，可调用 .backward() 反向传播。
        """
        # 攻击环境预设：确保模型处于评估模式，关闭 Dropout 与 BatchNorm 的训练时统计更新，
        # 消除随机性，保证多次计算损失时梯度方向稳定、可复现。
        self.model.eval()

        # 前向传播：获取模型对当前输入的原始 logits（未经 Softmax 的 1000 维输出）
        output = self.model(input_tensor)  # shape: (1, 1000)

        # 构造目标标签张量，确保数据类型与设备均与模型输出对齐
        target = torch.tensor([target_id], dtype=torch.long, device=self.device)

        # 使用交叉熵损失衡量当前输出与目标类别的差距
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)

        return loss

    def extract_gradient(self, loss: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        通过反向传播提取输入域梯度（Data Gradient）。

        为什么针对 target_id 计算损失：
        在非定向攻击中，我们最大化真实标签的损失，迫使模型"远离"正确答案。
        在定向攻击（Targeted Attack）中，策略相反：我们最小化目标类别的损失，
        相当于强迫模型把输入图像"认错"成我们指定的类别（如把狗错认为鸡）。
        CrossEntropyLoss(output, target_id) 越小，模型对 target_id 的置信度越高，
        因此通过降低该损失，即可实现定向误导。

        loss.backward() 的链式法则原理：
        1. 损失函数 L 对模型最后一层输出（logits）求导：∂L/∂output；
        2. 通过神经网络各层逐层反向传播（Chain Rule）：∂L/∂W, ∂L/∂X_layer；
        3. 最终传播到输入层，得到 ∂L/∂X_input（即 Data Gradient）。
        这个梯度向量的每一个元素都告诉了我们：对应位置的像素值应该如何微调，
        才能最有效地降低损失（也就是让模型更确信输入属于 target_id）。
        FGSM 正是利用这个梯度的符号（sign）来构造扰动方向。

        Args:
            loss: 定向攻击损失标量（由 compute_targeted_loss 输出）。
            input_tensor: 已开启梯度追踪的输入张量，反向传播后其 .grad 属性会被填充。

        Returns:
            torch.Tensor: 输入域梯度 data_grad，形状与 input_tensor 相同，位于 self.device。
        """
        # 安全性与显存优化：在反向传播前彻底清空残余梯度。
        # 若不清零，前一次推理或攻击残留的梯度会累加到本次，导致攻击方向偏离真实梯度。
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()

        # 反向传播：PyTorch Autograd 自动应用链式法则，
        # 从损失节点回溯至输入节点，计算并填充 input_tensor.grad。
        loss.backward()

        # 提取数据域梯度：.grad.data 直接获取底层张量数据，避免携带计算图元数据，节省显存。
        data_grad = input_tensor.grad.data

        return data_grad

    def generate_targeted_adversarial(
        self,
        original_tensor: torch.Tensor,
        target_id: int,
        epsilon: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于 FGSM 算法在像素空间生成定向对抗样本。

        核心改进：将攻击从归一化空间迁移到像素空间 [0, 1]，
        在像素空间开启梯度追踪并执行扰动，彻底避免归一化空间 clamp 导致的信息丢失。

        核心公式（像素空间）：
            X_adv_pixel = clamp( X_pixel - epsilon * sign( grad_pixel ), 0, 1 )
        其中 grad_pixel = d(Loss) / d(X_pixel)。

        Args:
            original_tensor: 原始图像预处理张量，形状 (1, C, H, W)。
            target_id: 目标类别索引（如 7 代表母鸡）。
            epsilon: 扰动强度（像素空间），控制每个像素的最大改变量（典型值 0.01 ~ 0.1）。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - adv_tensor: 对抗样本张量（归一化空间），供 predict() 直接使用。
                - perturbation: 纯扰动张量（像素空间），用于热力图可视化。
                - pixel_grad: 像素空间梯度，可用于进一步分析。
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # 1. 反归一化到像素空间 [0, 1]
        pixel_input = original_tensor.clone().detach().to(self.device)
        pixel_input = pixel_input * std + mean

        # 2. 在像素空间开启梯度追踪（攻击在此空间执行，避免归一化空间 clamp 扭曲）
        pixel_input.requires_grad_(True)

        # 3. 重新归一化后送入模型前向传播
        norm_input = (pixel_input - mean) / std
        self.model.eval()
        output = self.model(norm_input)

        # 4. 计算定向损失：最小化目标类别的交叉熵
        target = torch.tensor([target_id], dtype=torch.long, device=self.device)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)

        # 5. 反向传播到像素空间，得到像素域梯度
        self.model.zero_grad()
        loss.backward()
        pixel_grad = pixel_input.grad.data

        # 6. 像素空间 FGSM：定向攻击用减号（最小化目标损失）
        perturbation = epsilon * pixel_grad.sign()
        pixel_adv = pixel_input - perturbation

        # 7. 在像素空间执行合法的 [0, 1] 截断
        pixel_adv = torch.clamp(pixel_adv, 0.0, 1.0)

        # 8. 重新归一化，供模型后续推理
        adv_tensor = (pixel_adv - mean) / std

        return adv_tensor, perturbation, pixel_grad

    def generate_targeted_pgd(
        self,
        original_tensor: torch.Tensor,
        target_id: int,
        epsilon: float,
        alpha: Optional[float] = None,
        num_iter: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于 PGD（Projected Gradient Descent）的定向对抗样本生成。

        PGD 是 FGSM 的多步迭代版本，每步只走一小步（步长 alpha），
        然后将结果投影回原始样本的 epsilon 邻域内，最后做像素截断。
        对于高置信度或大类间攻击（如 banana → cock），PGD 成功率远高于单步 FGSM。

        算法流程：
            X_0 = X
            for t = 1 to num_iter:
                X_t = X_{t-1} - alpha * sign( grad(X_{t-1}) )
                X_t = clip_{epsilon}( X_t, X )   # 投影回 epsilon 邻域
                X_t = clamp( X_t, 0, 1 )         # 截断到合法像素范围
            return X_{num_iter}

        Args:
            original_tensor: 原始图像预处理张量，形状 (1, C, H, W)。
            target_id: 目标类别索引。
            epsilon: 最大扰动半径（像素空间 L-inf 范数）。
            alpha: 每步步长，默认 epsilon / 4。
            num_iter: 迭代次数，默认 10。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - adv_tensor: 对抗样本（归一化空间）。
                - total_perturbation: 总扰动量（像素空间）。
        """
        if alpha is None:
            alpha = epsilon / 4.0

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # 反归一化到像素空间
        pixel_orig = original_tensor.clone().detach().to(self.device)
        pixel_orig = pixel_orig * std + mean  # [0, 1] 空间

        # 初始化对抗样本
        pixel_adv = pixel_orig.clone()

        for i in range(num_iter):
            pixel_adv.requires_grad_(True)

            # 归一化后送入模型
            norm_input = (pixel_adv - mean) / std
            output = self.model(norm_input)

            target = torch.tensor([target_id], dtype=torch.long, device=self.device)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, target)

            self.model.zero_grad()
            loss.backward()
            grad = pixel_adv.grad.data

            # 单步更新：像素空间梯度下降
            pixel_adv = pixel_adv - alpha * grad.sign()

            # 投影回 epsilon 邻域：限制与原始样本的距离
            perturbation = torch.clamp(pixel_adv - pixel_orig, -epsilon, epsilon)
            pixel_adv = pixel_orig + perturbation

            # 截断到合法像素范围
            pixel_adv = torch.clamp(pixel_adv, 0.0, 1.0).detach()

        total_perturbation = pixel_adv - pixel_orig
        adv_tensor = (pixel_adv - mean) / std
        return adv_tensor, total_perturbation

    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """
        将经过 ImageNet 归一化的张量还原为可显示的 PIL Image。

        预处理时 torchvision 执行了 Normalize(mean, std)：
            x_norm = (x - mean) / std
        因此反归一化需要：
            x = x_norm * std + mean
        然后再从 (C, H, W) 转置为 (H, W, C)，并映射到 [0, 255]。

        Args:
            tensor: 归一化后的图像张量，形状 (1, C, H, W) 或 (C, H, W)，值域经归一化后可能超出 [0, 1]。

        Returns:
            Image.Image: 反归一化后的 PIL RGB 图像。
        """
        # ImageNet 官方归一化参数
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)

        # 去除 Batch 维度（若存在）
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # 反归一化：x = x_norm * std + mean
        tensor = tensor * std + mean

        # 截断到合法像素范围 [0, 1]，再映射到 [0, 255]
        tensor = torch.clamp(tensor, 0.0, 1.0)
        arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        return Image.fromarray(arr)
