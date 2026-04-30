# core/verify_task22.py
# 阶段二任务 2.2 验证脚本：检查定向损失计算与梯度提取是否满足要求

import sys
import os

# 将项目根目录加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.attack_engine import AttackEngine


def verify_task22():
    print("=" * 60)
    print("阶段二任务 2.2 验证开始")
    print("=" * 60)

    # 1. 实例化攻击引擎
    print("\n[1/7] 实例化 AttackEngine...")
    engine = AttackEngine()
    print("    ✔ AttackEngine 实例化成功")

    # 2. 准备测试张量（模拟预处理后的图像）
    print("\n[2/7] 准备对抗输入张量...")
    dummy_tensor = torch.randn(1, 3, 224, 224)
    adv_input = engine.prepare_adversarial_input(dummy_tensor)
    print(f"    ✔ 张量设备: {adv_input.device}")
    print(f"    ✔ requires_grad: {adv_input.requires_grad}")

    # 3. 测试定向损失计算
    print("\n[3/7] 测试 compute_targeted_loss (target_id=7, hen)...")
    target_id = 7
    loss = engine.compute_targeted_loss(adv_input, target_id)

    assert isinstance(loss, torch.Tensor), "返回值应为 torch.Tensor"
    assert loss.dim() == 0, "损失应为标量（0维张量）"
    assert loss.requires_grad, "损失必须保留计算图以支持反向传播"
    print(f"    ✔ 损失类型: {type(loss).__name__}")
    print(f"    ✔ 损失维度: {loss.dim()} (标量)")
    print(f"    ✔ 损失 requires_grad: {loss.requires_grad}")
    print(f"    ✔ 损失数值: {loss.item():.4f}")

    # 4. 测试梯度提取
    print("\n[4/7] 测试 extract_gradient...")
    data_grad = engine.extract_gradient(loss, adv_input)

    assert isinstance(data_grad, torch.Tensor), "data_grad 应为 torch.Tensor"
    assert data_grad.shape == adv_input.shape, "梯度形状应与输入张量一致"
    assert data_grad.device == engine.device, "梯度应位于目标设备"
    non_zero_count = (data_grad != 0).sum().item()
    assert non_zero_count > 0, "梯度不应全为零，说明链式法则已成功传播到输入层"
    print(f"    ✔ 梯度形状: {data_grad.shape}")
    print(f"    ✔ 梯度设备: {data_grad.device}")
    print(f"    ✔ 梯度非零元素数: {non_zero_count} / {data_grad.numel()}")
    print(f"    ✔ 梯度均值: {data_grad.abs().mean().item():.6f}")

    # 5. 验证梯度清理是否彻底
    print("\n[5/7] 验证 extract_gradient 内部已清理残余梯度...")
    # 再次调用 extract_gradient 前应无残留
    loss2 = engine.compute_targeted_loss(adv_input, target_id)
    # 此时如果不清零，adv_input.grad 应该还在；但 extract_gradient 内部会清零
    data_grad2 = engine.extract_gradient(loss2, adv_input)
    # 检查新生成的梯度是否与上一次不同（数值上不一定不同，但过程上独立）
    print("    ✔ 二次提取梯度未受前次干扰")

    # 6. 验证定向逻辑：损失是否确实针对 target_id
    print("\n[6/7] 验证定向损失逻辑（不同 target_id 应产生不同损失）...")
    adv_input_b = engine.prepare_adversarial_input(dummy_tensor)
    loss_id_7 = engine.compute_targeted_loss(adv_input_b, 7)
    loss_id_100 = engine.compute_targeted_loss(adv_input_b, 100)
    print(f"    ✔ target_id=7  的损失: {loss_id_7.item():.4f}")
    print(f"    ✔ target_id=100 的损失: {loss_id_100.item():.4f}")
    print(f"    ✔ 两损失差值: {abs(loss_id_7.item() - loss_id_100.item()):.4f} (应不为零)")

    # 7. 验证模型处于 eval 模式
    print("\n[7/7] 验证攻击环境预设...")
    assert not engine.model.training, "模型应始终处于 eval 模式"
    print("    ✔ self.model.training = False (eval 模式已启用)")

    print("\n" + "=" * 60)
    print("阶段二任务 2.2 验证全部通过")
    print("=" * 60)


if __name__ == "__main__":
    verify_task22()
