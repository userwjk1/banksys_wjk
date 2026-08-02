"""测试 src/analysis.py — 数据分析逻辑."""

import pytest
import pandas as pd
import numpy as np
from src.analysis import (
    compute_summary_stats,
    compute_subscribe_rate,
    compute_correlation_matrix,
)


@pytest.fixture
def sample_df():
    """构建模拟的银行营销预编码数据."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age": np.random.randint(20, 70, n),
        "job": np.random.randint(0, 5, n),
        "marital": np.random.randint(0, 4, n),
        "education": np.random.randint(0, 5, n),
        "default": np.random.randint(0, 2, n),
        "housing": np.random.randint(0, 2, n),
        "loan": np.random.randint(0, 2, n),
        "campaign": np.random.randint(1, 10, n),
        "pdays": np.random.choice([999, 0, 3, 6], n),
        "previous": np.random.randint(0, 5, n),
        "emp_var_rate": np.random.uniform(-3, 2, n),
        "cons_price_index": np.random.uniform(89, 97, n),
        "cons_conf_index": np.random.uniform(-45, -30, n),
        "lending_rate3m": np.random.uniform(0.5, 5.5, n),
        "nr_employed": np.random.uniform(4900, 5300, n),
    })


class TestComputeSummaryStats:
    """测试 compute_summary_stats."""

    def test_returns_dataframe(self, sample_df):
        """返回 DataFrame 且含关键统计列."""
        stats = compute_summary_stats(sample_df)
        assert isinstance(stats, pd.DataFrame)

    def test_contains_statistics(self, sample_df):
        """含均值、中位数、标准差等统计量."""
        stats = compute_summary_stats(sample_df)
        # 应至少包含常见统计量
        stat_names = stats.index.tolist()
        expected = ["mean", "std", "min", "25%", "50%", "75%", "max"]
        for stat in expected:
            assert stat in stat_names, f"缺少统计量: {stat}"

    def test_empty_dataframe(self):
        """空 DataFrame 返回空 describe."""
        df_empty = pd.DataFrame()
        stats = compute_summary_stats(df_empty)
        assert isinstance(stats, pd.DataFrame)


class TestComputeSubscribeRate:
    """测试 compute_subscribe_rate."""

    def test_binary_target(self):
        """二分类目标列返回正确的认购率."""
        y = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])
        rate = compute_subscribe_rate(y)
        assert 0.0 <= rate <= 1.0
        assert rate == pytest.approx(0.5)

    def test_all_yes(self):
        """全部认购返回 1.0."""
        y = pd.Series([1, 1, 1])
        assert compute_subscribe_rate(y) == 1.0

    def test_all_no(self):
        """全部未认购返回 0.0."""
        y = pd.Series([0, 0, 0])
        assert compute_subscribe_rate(y) == 0.0

    def test_empty_series(self):
        """空 Series 返回 0.0."""
        y = pd.Series([], dtype=int)
        assert compute_subscribe_rate(y) == 0.0


class TestComputeCorrelationMatrix:
    """测试 compute_correlation_matrix."""

    def test_returns_dataframe(self, sample_df):
        """返回方形的相关性矩阵."""
        corr = compute_correlation_matrix(sample_df)
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape[0] == corr.shape[1]

    def test_diagonal_is_one(self, sample_df):
        """相关性矩阵对角线为 1."""
        corr = compute_correlation_matrix(sample_df)
        for col in corr.columns:
            assert corr.loc[col, col] == pytest.approx(1.0)

    def test_excludes_non_numeric(self):
        """自动排除非数值列."""
        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0],
            "num2": [4.0, 5.0, 6.0],
            "str_col": ["a", "b", "c"],
        })
        corr = compute_correlation_matrix(df)
        assert "str_col" not in corr.columns
        assert "num1" in corr.columns
        assert "num2" in corr.columns
