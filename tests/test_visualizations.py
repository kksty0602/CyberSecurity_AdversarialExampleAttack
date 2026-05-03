import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from components.visualizations import (
    plot_perturbation_heatmap,
    plot_confidence_bar_chart,
    plot_pgd_convergence_curve,
)


def test_heatmap_returns_figure():
    perturbation = torch.randn(1, 3, 224, 224) * 0.01
    fig = plot_perturbation_heatmap(perturbation)
    assert fig is not None


def test_bar_chart_returns_figure():
    fig = plot_confidence_bar_chart("dog", 87.5, "chicken", 62.3)
    assert fig is not None


def test_pgd_curve_returns_figure():
    fig = plot_pgd_convergence_curve([5.0, 15.0, 35.0, 55.0, 62.3])
    assert fig is not None
