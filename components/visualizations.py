"""
components/visualizations.py
封装 Matplotlib 图表生成函数，供 Attack Tab 和 Defense Tab 共用。
"""

from typing import List, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def plot_perturbation_heatmap(perturbation: torch.Tensor) -> plt.Figure:
    """
    绘制扰动热力图。

    Args:
        perturbation: 扰动张量，形状 (1, C, H, W) 或 (C, H, W)，像素空间。

    Returns:
        plt.Figure: Matplotlib Figure 对象，可直接传给 st.pyplot()。
    """
    if perturbation.dim() == 4:
        perturbation = perturbation.squeeze(0)

    # 取三个通道的平均绝对值，得到单通道热力图
    heatmap = perturbation.abs().mean(dim=0).cpu().numpy()  # (H, W)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(heatmap, cmap="hot")
    ax.axis("off")
    ax.set_title("Perturbation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def plot_confidence_bar_chart(
    original_name: str,
    original_conf: float,
    target_name: str,
    target_conf: float,
) -> plt.Figure:
    """
    绘制原始 Top-1 类别与目标类别的置信度对比柱状图。

    Args:
        original_name: 原始 Top-1 类别名称。
        original_conf: 原始 Top-1 置信度（百分比，如 87.5）。
        target_name: 目标类别名称。
        target_conf: 目标类别在对抗样本上的置信度（百分比）。

    Returns:
        plt.Figure: Matplotlib Figure 对象。
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    categories = [f"原始: {original_name}", f"目标: {target_name}"]
    values = [original_conf, target_conf]
    colors = ["steelblue", "coral"]

    bars = ax.bar(categories, values, color=colors, width=0.5)
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Confidence Comparison")
    ax.set_ylim(0, 100)

    # 在柱顶标注数值
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    return fig


def plot_pgd_convergence_curve(target_confs: List[float]) -> plt.Figure:
    """
    绘制 PGD 攻击过程中目标类别置信度的迭代变化曲线。

    Args:
        target_confs: 每轮迭代后目标类别的置信度列表（百分比）。

    Returns:
        plt.Figure: Matplotlib Figure 对象。
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    iterations = list(range(1, len(target_confs) + 1))
    ax.plot(iterations, target_confs, marker="o", color="coral", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Target Confidence (%)")
    ax.set_title("PGD Convergence Curve")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig
