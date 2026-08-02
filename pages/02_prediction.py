"""功能二:在线预测页面 — 点选表单输入客户特征,预测认购结果."""

import os
import streamlit as st
import pandas as pd

from src.data_loader import load_data, CATEGORICAL_COLS
from src.predict import predict_single, validate_input, NUMERIC_RANGES

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(page_title="在线预测", page_icon="🔮", layout="wide")

st.title("🔮 在线预测 — 客户认购预测")
st.markdown("---")

MODEL_PATH = "models/model.joblib"
DATA_PATH = "data/train.csv"


@st.cache_resource
def load_model_cached(path: str):
    """缓存加载预测模型."""
    from src.train import load_model
    return load_model(path)


@st.cache_data
def load_reference_data(path: str) -> pd.DataFrame:
    """加载训练数据以提取分类选项和数值范围."""
    return load_data(path)


# ---------------------------------------------------------------------------
# 模型可用性检查
# ---------------------------------------------------------------------------
if not os.path.isfile(MODEL_PATH):
    st.error(
        "⚠️ **模型尚未训练，请先运行训练脚本**\n\n"
        "```bash\n"
        "python -m src.train --data-path data/train.csv --model-path models/model.joblib\n"
        "```"
    )
    st.stop()

try:
    model = load_model_cached(MODEL_PATH)
    ref_df = load_reference_data(DATA_PATH)
except Exception as e:
    st.error(f"模型或数据加载失败: {e}")
    st.stop()

# 获取模型期望的输入特征名(排除 duration 和 subscribe)
try:
    model_features = list(model.feature_names_in_)
except AttributeError:
    model_features = [c for c in ref_df.columns
                      if c not in ("id", "duration", "subscribe")]

# ---------------------------------------------------------------------------
# 输入表单
# ---------------------------------------------------------------------------
st.subheader("📝 客户特征输入")

# 数值特征 — 默认值取训练数据的中位数
numeric_features = [f for f in model_features
                    if f not in CATEGORICAL_COLS]

# 分类特征 — 选项从训练数据取值中提取
cat_features_in_model = [f for f in model_features
                         if f in CATEGORICAL_COLS]

st.markdown("#### 基本信息")

col1, col2, col3 = st.columns(3)

input_data = {}

# --- 数值特征 ---
for i, feat in enumerate(numeric_features):
    col = [col1, col2, col3][i % 3]
    lo, hi = NUMERIC_RANGES.get(feat, (0, 100000))
    default_val = float(ref_df[feat].median()) if feat in ref_df.columns else float((lo + hi) / 2)

    if feat == "age":
        input_data[feat] = col.slider(
            "年龄", lo, hi, int(default_val), step=1,
        )
    elif feat == "campaign":
        input_data[feat] = col.slider(
            "营销接触次数", lo, hi, int(default_val), step=1,
        )
    elif feat == "pdays":
        input_data[feat] = col.slider(
            "距上次联系天数(999=首次)", lo, hi, int(default_val), step=1,
        )
    elif feat == "previous":
        input_data[feat] = col.slider(
            "历史接触次数", lo, hi, int(default_val), step=1,
        )
    else:
        input_data[feat] = col.number_input(
            feat, lo, hi, value=default_val, step=0.01,
            format="%.2f",
        )

# --- 分类特征 ---
st.markdown("#### 客户属性")

cat_cols_per_row = 3
cat_rows = [
    cat_features_in_model[i:i + cat_cols_per_row]
    for i in range(0, len(cat_features_in_model), cat_cols_per_row)
]

for row_features in cat_rows:
    cols = st.columns(cat_cols_per_row)
    for j, feat in enumerate(row_features):
        options = sorted(ref_df[feat].dropna().unique().tolist()) \
            if feat in ref_df.columns else []
        default_idx = 0
        input_data[feat] = cols[j].selectbox(
            feat, options, index=default_idx,
        )

# ---------------------------------------------------------------------------
# 预测按钮
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🎯 预测")

predict_btn = st.button("🔍 开始预测", type="primary", use_container_width=True)

if predict_btn:
    # 输入校验
    valid, errors = validate_input(input_data, model_features)
    if not valid:
        for err in errors:
            st.error(f"❌ {err}")
    else:
        with st.spinner("模型推理中..."):
            try:
                result = predict_single(MODEL_PATH, input_data)
            except Exception as e:
                st.error(f"预测失败: {e}")
                st.stop()

        st.markdown("---")
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if result["prediction"] == 1:
                st.success(f"### ✅ {result['label']}")
                st.metric("认购概率", f"{result['probability']:.1%}")
            else:
                st.warning(f"### ❌ {result['label']}")
                st.metric("认购概率", f"{result['probability']:.1%}")

        with result_col2:
            st.markdown("#### 概率解读")
            prob = result["probability"]
            st.progress(prob)
            if prob >= 0.7:
                st.caption("🟢 极大概率认购,建议优先跟进")
            elif prob >= 0.4:
                st.caption("🟡 中等概率,建议进一步触达")
            else:
                st.caption("🔴 低概率,建议降低优先级")
