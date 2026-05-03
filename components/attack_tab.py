"""
components/attack_tab.py
Streamlit "攻击实验室" Tab 的完整 UI 与交互逻辑。
"""

from typing import Optional
import os

import streamlit as st
from PIL import Image

from core.attack_engine import AttackEngine
from components.visualizations import (
    plot_perturbation_heatmap,
    plot_confidence_bar_chart,
    plot_pgd_convergence_curve,
)
from utils.imagenet_labels import get_label_options, parse_label_option


def render_attack_tab(model: AttackEngine) -> None:
    """渲染攻击实验室 Tab。"""
    st.header("攻击实验室")

    # ------------------ 侧边栏控制区 ------------------
    with st.sidebar:
        st.subheader("攻击参数配置")

        # 图片输入：上传 或 选择示例
        upload_option = st.radio("图片来源", ["上传图片", "选择示例图"])
        image: Optional[Image.Image] = None

        if upload_option == "上传图片":
            uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "webp"])
            if uploaded is not None:
                image = Image.open(uploaded).convert("RGB")
        else:
            testset_dir = os.path.join(os.path.dirname(__file__), "..", "testset")
            if os.path.isdir(testset_dir):
                sample_files = [
                    f for f in os.listdir(testset_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                ]
                if sample_files:
                    selected = st.selectbox("选择示例图", sample_files)
                    image = Image.open(os.path.join(testset_dir, selected)).convert("RGB")
                else:
                    st.warning("testset 目录为空")
            else:
                st.warning("testset 目录不存在")

        # 攻击算法选择
        algorithm = st.radio("攻击算法", ["FGSM", "PGD"])

        # 目标类别
        label_options = get_label_options()
        target_option = st.selectbox("目标类别", label_options, index=7)
        target_id = parse_label_option(target_option)

        # epsilon
        epsilon = st.slider("epsilon 强度", 0.0, 0.1, 0.03, 0.01)

        # PGD 专属参数
        num_iter = 20
        if algorithm == "PGD":
            num_iter = st.slider("迭代次数", 5, 50, 20, 5)

        generate_btn = st.button("生成对抗样本", type="primary")

    # ------------------ 主展示区 ------------------
    if image is None:
        st.info("请在侧边栏上传图片或选择示例图")
        return

    # 预处理并获取原始预测
    original_tensor = model.preprocess(image)
    original_result = model.predict(original_tensor)
    original_name = original_result["topk_names"][0]
    original_conf = original_result["topk_confs"][0]

    st.markdown(f"**原始预测: {original_name}  {original_conf:.2f}%**")

    # 动态提示（基于原始置信度）
    if original_conf >= 80:
        st.warning("模型对原图预测非常确定，建议使用 PGD 算法或 epsilon >= 0.05")
    elif original_conf >= 50:
        st.info("模型对原图有一定置信度，FGSM 建议 epsilon >= 0.03")
    else:
        st.success("模型对原图预测不确定，FGSM 小步长即可攻击成功")

    # 显示原始图像与 Top-5
    col_orig, col_heat, col_adv = st.columns(3)

    with col_orig:
        st.subheader("原始图像")
        st.image(image, use_container_width=True)
        st.markdown("**Top-5 预测：**")
        for name, conf in zip(original_result["topk_names"], original_result["topk_confs"]):
            st.write(f"- {name}: {conf:.2f}%")

    # 执行攻击
    if generate_btn:
        with st.spinner("正在计算梯度并生成对抗样本..."):
            if algorithm == "FGSM":
                adv_tensor, perturbation, _ = model.generate_targeted_adversarial(
                    original_tensor, target_id, epsilon
                )
                pgd_history = None
            else:
                adv_tensor, perturbation, pgd_history = model.generate_targeted_pgd_with_history(
                    original_tensor, target_id, epsilon, num_iter=num_iter
                )

        adv_result = model.predict(adv_tensor)
        adv_name = adv_result["topk_names"][0]
        adv_conf = adv_result["topk_confs"][0]

        # 查找目标类别的索引和置信度
        target_confs = adv_result["topk_confs"]
        target_ids = adv_result["topk_ids"]
        target_label_name = target_option.split(": ")[1]
        target_conf = 0.0
        if target_id in target_ids:
            idx = target_ids.index(target_id)
            target_conf = target_confs[idx]

        # 保存到 session_state 供防御 Tab 使用
        adv_image = model.tensor_to_image(adv_tensor)
        st.session_state["adv_image"] = adv_image
        st.session_state["adv_result"] = adv_result
        st.session_state["original_result"] = original_result

        # 噪声热力图
        with col_heat:
            st.subheader("噪声热力图")
            fig_heat = plot_perturbation_heatmap(perturbation)
            st.pyplot(fig_heat)
            max_perturbation = perturbation.abs().max().item()
            st.caption(f"最大扰动值: {max_perturbation:.4f} (epsilon={epsilon})")

        # 对抗样本与预测结果
        with col_adv:
            st.subheader("对抗样本")
            st.image(adv_image, use_container_width=True)
            st.markdown(f"**对抗预测: {adv_name}  {adv_conf:.2f}%**")
            st.markdown("**Top-5 预测：**")
            for name, conf in zip(adv_result["topk_names"], adv_result["topk_confs"]):
                st.write(f"- {name}: {conf:.2f}%")

        # 置信度对比柱状图
        st.subheader("置信度对比")
        fig_bar = plot_confidence_bar_chart(original_name, original_conf, target_label_name, target_conf)
        st.pyplot(fig_bar)

        # PGD 迭代曲线（仅 PGD 时显示）
        if pgd_history is not None:
            st.subheader("PGD 迭代收敛曲线")
            fig_curve = plot_pgd_convergence_curve(pgd_history)
            st.pyplot(fig_curve)
