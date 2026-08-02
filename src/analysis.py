"""数据分析模块 — 统计摘要、认购率、相关性等计算逻辑.

所有函数均为纯函数:输入 DataFrame/Series,输出计算结果。
方便单元测试,也方便 Streamlit 页面中带缓存调用。
"""

import pandas as pd


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """计算数值列的描述性统计.

    Args:
        df: 特征矩阵(数值型或混合型).

    Returns:
        统计摘要 DataFrame(类似 df.describe()).
    """
    if df.empty:
        return pd.DataFrame()
    return df.describe(include="all")


def compute_subscribe_rate(y: pd.Series) -> float:
    """计算认购率.

    Args:
        y: 目标列 Series(0/1),其中 1 表示认购.

    Returns:
        认购率(0.0 ~ 1.0).
    """
    if len(y) == 0:
        return 0.0
    return float(y.mean())


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """计算数值特征的相关性矩阵.

    Args:
        df: 特征矩阵.

    Returns:
        相关性矩阵(方阵,只含数值列).
    """
    numeric_df = df.select_dtypes(include=["number"])
    return numeric_df.corr()
