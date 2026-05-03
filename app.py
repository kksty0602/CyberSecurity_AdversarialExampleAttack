"""
app.py
Streamlit 主应用入口。
提供"攻击实验室"与"防御加固"两个 Tab，整合模型缓存与组件调用。
"""

import streamlit as st

from core.loadModel import get_adversarial_model
from components.attack_tab import render_attack_tab


def main():
    st.set_page_config(
        page_title="ResNet50 对抗攻击演示",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("ResNet50 对抗攻击与防御演示")
    st.caption("基于 PyTorch 预训练 ResNet50，实现 FGSM/PGD 定向对抗样本攻击与预处理防御")

    # 缓存加载模型（单例）
    with st.spinner("正在加载 ResNet50 模型..."):
        model = get_adversarial_model()

    # Tab 导航
    tab_attack, tab_defense = st.tabs(["攻击实验室", "防御加固"])

    with tab_attack:
        render_attack_tab(model)

    with tab_defense:
        st.header("防御加固")
        st.info("防御功能将在第二轮实现。请先在'攻击实验室'生成对抗样本。")


if __name__ == "__main__":
    main()
