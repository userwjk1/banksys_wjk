"""banksys_wjk — 银行营销数据交互分析与在线预测系统.

Streamlit 主入口,自动发现 pages/ 目录下的子页面。
"""

import streamlit as st

st.set_page_config(
    page_title="银行营销数据分析与预测系统",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🏦 银行营销系统")
st.sidebar.markdown("---")
st.sidebar.info(
    "**功能导航**\n\n"
    "📊 **数据分析** — 交互式探索银行营销数据\n\n"
    "🔮 **在线预测** — 输入客户特征预测认购结果"
)

st.title("🏦 银行营销数据分析与预测系统")
st.markdown("---")
st.markdown("### 👈 请从侧边栏选择功能页面")
st.markdown(
    """
    - **📊 数据分析**: 查看客户特征分布、订阅转化率、相关性热力图等交互式图表
    - **🔮 在线预测**: 基于训练好的 ML 模型,输入客户特征即时预测认购概率
    """
)
