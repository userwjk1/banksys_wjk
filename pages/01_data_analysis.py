"""功能一:数据分析交互页面 — 银行营销数据探索与可视化."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from src.data_loader import load_data, preprocess, CATEGORICAL_COLS
from src.analysis import compute_summary_stats, compute_subscribe_rate, compute_correlation_matrix

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(page_title="数据分析", page_icon="📊", layout="wide")

st.title("📊 数据分析 — 银行营销数据探索")
st.markdown("---")

# ---------------------------------------------------------------------------
# 数据加载与缓存
# ---------------------------------------------------------------------------
DATA_PATH = "data/train.csv"


@st.cache_data
def load_cached_data(path: str) -> pd.DataFrame:
    """缓存加载原始数据."""
    return load_data(path)


@st.cache_data
def load_cached_processed_data(path: str) -> tuple[pd.DataFrame, pd.Series | None]:
    """缓存加载预处理后的数据."""
    raw = load_data(path)
    return preprocess(raw, target_col="subscribe", exclude_duration=True)


try:
    raw_df = load_cached_data(DATA_PATH)
    processed_df, target = load_cached_processed_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"数据文件不存在: `{DATA_PATH}`。请确认数据已放入 `data/` 目录。")
    st.stop()

# ---------------------------------------------------------------------------
# 侧边栏 — 筛选
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 筛选条件")

# 年龄范围
age_min, age_max = int(raw_df["age"].min()), int(raw_df["age"].max())
age_range = st.sidebar.slider("年龄范围", age_min, age_max, (age_min, age_max))

# 职业选择
jobs = sorted(raw_df["job"].dropna().unique())
selected_jobs = st.sidebar.multiselect("职业(可多选)", jobs, default=jobs[:5])

# 教育程度
edu_levels = sorted(raw_df["education"].dropna().unique())
selected_edu = st.sidebar.multiselect("教育程度(可多选)", edu_levels, default=edu_levels[:3])

# 应用筛选
filtered = raw_df[
    (raw_df["age"] >= age_range[0])
    & (raw_df["age"] <= age_range[1])
    & (raw_df["job"].isin(selected_jobs))
    & (raw_df["education"].isin(selected_edu))
]

st.sidebar.markdown(f"筛选后样本数: **{len(filtered):,}**")

# ---------------------------------------------------------------------------
# Tab 1: 数据总览
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 数据总览", "📈 特征分布", "🔗 相关性分析", "📉 订阅转化分析",
])

with tab1:
    st.subheader("数据概览")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总样本数", f"{len(raw_df):,}")
    with col2:
        st.metric("特征数", raw_df.shape[1] - 1)
    with col3:
        subscribe_rate = compute_subscribe_rate(target) if target is not None else 0
        st.metric("认购率", f"{subscribe_rate:.1%}")
    with col4:
        st.metric("缺失值列数", raw_df.isna().sum().gt(0).sum())

    st.markdown("---")
    st.subheader("数据预览(前 10 行)")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("各列数据类型")
    dtypes_df = pd.DataFrame({
        "列名": raw_df.columns,
        "类型": raw_df.dtypes.astype(str).values,
        "非空数": raw_df.notna().sum().values,
        "缺失数": raw_df.isna().sum().values,
    })
    st.dataframe(dtypes_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: 特征分布
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("特征分布")

    col_left, col_right = st.columns([1, 3])

    with col_left:
        numeric_cols = raw_df.select_dtypes(include=["number"]).columns.tolist()
        # 排除 id 和 duration
        numeric_cols = [c for c in numeric_cols if c not in ("id", "duration")]
        cat_cols = [c for c in CATEGORICAL_COLS if c in raw_df.columns]

        chart_type = st.radio("选择特征类型", ["数值特征", "分类特征"])

    with col_right:
        if chart_type == "数值特征":
            selected_num = st.selectbox("选择数值列", numeric_cols)
            col_a, col_b = st.columns(2)
            with col_a:
                fig_hist = px.histogram(
                    filtered, x=selected_num, nbins=30,
                    title=f"{selected_num} 分布直方图",
                    marginal="box",
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            with col_b:
                stats = compute_summary_stats(filtered[[selected_num]])
                st.dataframe(stats, use_container_width=True)
        else:
            selected_cat = st.selectbox("选择分类列", cat_cols)
            value_counts = filtered[selected_cat].value_counts().reset_index()
            value_counts.columns = ["类别", "数量"]
            fig_bar = px.bar(
                value_counts, x="类别", y="数量",
                title=f"{selected_cat} 各类别分布",
                color="数量",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: 相关性分析
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("数值特征相关性热力图")

    numeric_filtered = filtered.select_dtypes(include=["number"])
    numeric_filtered = numeric_filtered.drop(columns=["id", "duration"], errors="ignore")

    if not numeric_filtered.empty:
        corr_matrix = compute_correlation_matrix(numeric_filtered)
        fig_heatmap = ff.create_annotated_heatmap(
            z=corr_matrix.values.round(3),
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            showscale=True,
        )
        fig_heatmap.update_layout(height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("无可用数值列进行相关性分析。")

# ---------------------------------------------------------------------------
# Tab 4: 订阅转化分析
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("各分类维度的订阅转化率")

    if target is not None:
        selected_cat_conv = st.selectbox(
            "选择维度查看订阅转化率",
            [c for c in CATEGORICAL_COLS if c in raw_df.columns],
        )

        conv_df = raw_df[[selected_cat_conv]].copy()
        conv_df["subscribe"] = target.values
        conv_rate = conv_df.groupby(selected_cat_conv)["subscribe"].mean().reset_index()
        conv_rate.columns = ["类别", "认购率"]
        conv_rate["认购率"] = conv_rate["认购率"] * 100

        fig_conv = px.bar(
            conv_rate, x="类别", y="认购率",
            title=f"{selected_cat_conv} 各分类认购率 (%)",
            color="认购率",
            text_auto=".1f",
        )
        fig_conv.update_traces(textposition="outside")
        st.plotly_chart(fig_conv, use_container_width=True)
    else:
        st.info("目标列不可用,无法计算转化率。")
