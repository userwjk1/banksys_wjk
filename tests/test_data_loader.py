"""测试 src/data_loader.py — 数据加载与预处理."""

import os
import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_data, preprocess, get_processed_data


@pytest.fixture
def sample_data_path(tmp_path):
    """创建包含缺失值和分类列的临时 CSV 供测试."""
    df = pd.DataFrame({
        "age": [30, 45, np.nan, 52, 38],
        "job": ["admin.", "blue-collar", "technician", np.nan, "admin."],
        "marital": ["married", "single", "married", "divorced", np.nan],
        "education": ["high.school", "basic.9y", np.nan, "professional.course", "high.school"],
        "default": ["no", "no", "no", "yes", "no"],
        "housing": ["yes", "no", "yes", np.nan, "no"],
        "loan": ["no", "yes", "no", "no", np.nan],
        "contact": ["cellular", "telephone", "cellular", np.nan, "cellular"],
        "month": ["may", "jun", "jul", "aug", np.nan],
        "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
        "duration": [120, 300, 0, 180, 240],
        "campaign": [1, 2, 3, 1, np.nan],
        "pdays": [999, 6, 999, 3, 999],
        "previous": [0, 1, 0, 2, 0],
        "poutcome": ["nonexistent", "failure", "success", np.nan, "nonexistent"],
        "emp_var_rate": [1.4, -1.8, 1.1, -0.1, 1.4],
        "cons_price_index": [90.81, 96.33, 89.67, 93.2, 92.0],
        "cons_conf_index": [-35.53, -40.58, -36.9, -41.0, np.nan],
        "lending_rate3m": [0.69, 4.05, 5.04, 3.27, 1.5],
        "nr_employed": [5219.74, 4974.79, 4947.02, 5203.33, 5022.61],
        "subscribe": ["no", "yes", "no", "yes", "no"],
        "id": [1, 2, 3, 4, 5],
    })
    path = tmp_path / "test_bank.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestLoadData:
    """测试 load_data 函数."""

    def test_load_valid_csv(self, sample_data_path):
        """正常加载:返回非空 DataFrame,列名与原始一致."""
        df = load_data(sample_data_path)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "subscribe" in df.columns
        assert "age" in df.columns
        assert "duration" in df.columns

    def test_load_file_not_found(self):
        """文件不存在:抛出 FileNotFoundError 并含路径提示."""
        with pytest.raises(FileNotFoundError, match="数据文件不存在"):
            load_data("nonexistent/file.csv")


class TestPreprocess:
    """测试 preprocess 函数."""

    def test_subscribe_encoded_to_binary(self, sample_data_path):
        """目标列 subscribe yes/no → 1/0."""
        df = load_data(sample_data_path)
        processed, target = preprocess(df, target_col="subscribe")
        assert target is not None
        assert set(target.unique()).issubset({0, 1})

    def test_missing_values_filled(self, sample_data_path):
        """缺失值按策略填充:数值列中位数,分类列众数."""
        df = load_data(sample_data_path)
        processed, _ = preprocess(df, target_col="subscribe")
        assert not processed.isnull().any().any()

    def test_duration_excluded_for_training(self, sample_data_path):
        """训练模式:排除 duration 列."""
        df = load_data(sample_data_path)
        processed, _ = preprocess(df, target_col="subscribe", exclude_duration=True)
        assert "duration" not in processed.columns

    def test_categorical_encoded(self, sample_data_path):
        """分类特征被编码为数值."""
        df = load_data(sample_data_path)
        processed, _ = preprocess(df, target_col="subscribe")
        categorical_cols = ["job", "marital", "education", "default",
                            "housing", "loan", "contact", "month", "day_of_week",
                            "poutcome"]
        for col in categorical_cols:
            if col in processed.columns:
                assert processed[col].dtype in [np.int32, np.int64, np.float64]


class TestGetProcessedData:
    """测试 get_processed_data 便捷函数."""

    def test_returns_X_y(self, sample_data_path):
        """返回特征矩阵 X 和目标 y."""
        X, y = get_processed_data(sample_data_path, target_col="subscribe")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert "duration" not in X.columns
        assert "subscribe" not in X.columns
        assert set(y.unique()).issubset({0, 1})

    def test_no_target_col_returns_only_X(self, sample_data_path):
        """数据无目标列时只返回 X,y 为 None."""
        df = load_data(sample_data_path)
        df_no_target = df.drop(columns=["subscribe"])
        no_target_path = os.path.join(
            os.path.dirname(sample_data_path), "no_target.csv"
        )
        df_no_target.to_csv(no_target_path, index=False)
        X, y = get_processed_data(no_target_path)
        assert isinstance(X, pd.DataFrame)
        assert y is None
