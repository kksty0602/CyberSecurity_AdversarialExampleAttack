"""
components/defense_tab.py
Streamlit "防御加固" Tab 的完整 UI 与交互逻辑。
"""

from typing import Optional

import streamlit as st
from PIL import Image

from core.attack_engine import AttackEngine
from core.defense_engine import DefenseEngine
from components.visualizations import plot_confidence_bar_chart


def render_defense_tab(model: AttackEngine) -> None:
    """渲染防御加固 Tab。"""
    st.header("防御加固")

    defense_engine = DefenseEngine()

    # ------------------ 输入来源 ------------------
    input_option = st.radio(
        "对抗样本来源",
        ["使用攻击实验室生成的样本", "上传对抗样本"],
        horizontal=True,
    )

    adv_image: Optional[Image.Image] = None
    adv_result = None
    original_result = None

    if input_option == "使用攻击实验室生成的样本":
        if "adv_image" in st.session_state:
            adv_image = st.session_state["adv_image"]
            adv_result = st.session_state.get("adv_result")
            original_result = st.session_state.get("original_result")
        else:
            st.warning("尚未生成对抗样本，请先在'攻击实验室'执行攻击。")
            return
    else:
        uploaded = st.file_uploader("上传对抗样本图片", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            adv_image = Image.open(uploaded).convert("RGB")
        else:
            st.info("请上传对抗样本图片")
            return

    # ------------------ 侧边栏控制区 ------------------
    with st.sidebar:
        st.subheader("防御参数配置")
        defense_method = st.radio("防御方法", ["高斯模糊", "JPEG 压缩"])

        if defense_method == "高斯模糊":
            sigma = st.slider("高斯核标准差", 0.5, 5.0, 1.0, 0.5)
        else:
            quality = st.slider("JPEG 质量因子", 10, 100, 75, 5)

        apply_btn = st.button("应用防御处理", type="primary")

    # ------------------ 主展示区 ------------------
    # 防御前预测（如果 session_state 中有则用那里的，否则重新计算）
    if adv_result is None and adv_image is not None:
        adv_tensor = model.preprocess(adv_image)
        adv_result = model.predict(adv_tensor)

    if adv_result is None:
        st.error("无法获取对抗样本的预测结果")
        return

    adv_name = adv_result["topk_names"][0]
    adv_conf = adv_result["topk_confs"][0]

    st.markdown(f"**防御前 Top-1: {adv_name}  {adv_conf:.2f}%**")

    col_before, col_after = st.columns(2)

    with col_before:
        st.subheader("对抗样本（防御前）")
        st.image(adv_image, use_container_width=True)
        st.markdown("**Top-5 预测：**")
        for name, conf in zip(adv_result["topk_names"], adv_result["topk_confs"]):
            st.write(f"- {name}: {conf:.2f}%")

    # 应用防御
    if apply_btn:
        with st.spinner("正在应用防御处理..."):
            if defense_method == "高斯模糊":
                defended_image = defense_engine.gaussian_defense(adv_image, sigma)
            else:
                defended_image = defense_engine.jpeg_defense(adv_image, quality)

        defended_tensor = model.preprocess(defended_image)
        defended_result = model.predict(defended_tensor)
        defended_name = defended_result["topk_names"][0]
        defended_conf = defended_result["topk_confs"][0]

        # 判断防御是否成功（恢复为原始预测）
        original_name = original_result["topk_names"][0] if original_result else None
        is_success = original_name is not None and defended_name == original_name

        with col_after:
            st.subheader("防御处理后")
            st.image(defended_image, use_container_width=True)
            st.markdown(f"**防御后 Top-1: {defended_name}  {defended_conf:.2f}%**")

            if is_success:
                st.success(f"防御成功：模型已恢复为 {original_name}")
            else:
                st.error("防御失败：模型仍被误导")

            st.markdown("**Top-5 预测：**")
            for name, conf in zip(defended_result["topk_names"], defended_result["topk_confs"]):
                st.write(f"- {name}: {conf:.2f}%")

        # 防御前后置信度对比
        if original_result:
            st.subheader("防御效果对比")
            original_name_str = original_result["topk_names"][0]
            original_conf = original_result["topk_confs"][0]
            fig = plot_confidence_bar_chart(
                original_name_str, original_conf,
                adv_name, adv_conf,
            )
            st.pyplot(fig)
