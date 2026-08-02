"""测试 src/predict.py — 模型加载与预测."""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from src.predict import predict_single, validate_input
from src.train import save_model

# 银行营销数据的分类特征
CAT_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]


@pytest.fixture
def simple_model(tmp_path):
    """训练一个可处理混合数据的二分类模型用于测试."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "age": np.random.randint(20, 70, n),
        "campaign": np.random.randint(1, 5, n),
        "pdays": np.random.choice([999, 3, 6], n),
        "previous": np.random.randint(0, 3, n),
        "emp_var_rate": np.random.uniform(-3, 2, n),
        "cons_price_index": np.random.uniform(89, 97, n),
        "cons_conf_index": np.random.uniform(-45, -30, n),
        "lending_rate3m": np.random.uniform(0.5, 5.5, n),
        "nr_employed": np.random.uniform(4900, 5300, n),
        "job": np.random.choice(["admin.", "technician", "blue-collar"], n),
        "marital": np.random.choice(["married", "single"], n),
        "education": np.random.choice(["high.school", "university.degree"], n),
        "default": np.random.choice(["no", "yes"], n, p=[0.8, 0.2]),
        "housing": np.random.choice(["yes", "no"], n),
        "loan": np.random.choice(["no", "yes"], n),
        "contact": np.random.choice(["cellular", "telephone"], n),
        "month": np.random.choice(["may", "jun", "jul"], n),
        "day_of_week": np.random.choice(["mon", "tue", "wed"], n),
        "poutcome": np.random.choice(["nonexistent", "failure", "success"], n),
    })
    y = np.random.choice([0, 1], n)

    numeric_features = [c for c in X.columns if c not in CAT_FEATURES]
    cat_features_present = [c for c in CAT_FEATURES if c in X.columns]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_features_present),
    ])

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    pipe.fit(X, y)

    model_path = str(tmp_path / "test_model.joblib")
    save_model(pipe, model_path)
    return model_path, X.columns.tolist()


class TestValidateInput:
    """测试 validate_input 输入校验."""

    def test_valid_input(self):
        """合理输入无异常."""
        data = {"age": 35, "campaign": 2, "job": "admin."}
        valid, errors = validate_input(data, feature_names=["age", "campaign", "job"])
        assert valid is True
        assert errors == []

    def test_age_out_of_range(self):
        """age 超出合理范围报错."""
        data = {"age": 500, "job": "admin."}
        valid, errors = validate_input(data, feature_names=["age", "job"])
        assert valid is False
        assert any("age" in e.lower() for e in errors)

    def test_negative_campaign(self):
        """campaign 负数报错."""
        data = {"campaign": -1, "age": 35}
        valid, errors = validate_input(data, feature_names=["age", "campaign"])
        assert valid is False
        assert any("campaign" in e.lower() for e in errors)


class TestPredictSingle:
    """测试 predict_single 单样本预测."""

    def test_returns_dict(self, simple_model):
        """返回预测结果字典."""
        model_path, feature_names = simple_model
        input_data = {
            "age": 40, "campaign": 2, "pdays": 999, "previous": 0,
            "emp_var_rate": 1.4, "cons_price_index": 93.0,
            "cons_conf_index": -38.0, "lending_rate3m": 2.5,
            "nr_employed": 5100.0,
            "job": "admin.", "marital": "married", "education": "high.school",
            "default": "no", "housing": "yes", "loan": "no",
            "contact": "cellular", "month": "may", "day_of_week": "mon",
            "poutcome": "nonexistent",
        }
        result = predict_single(model_path, input_data)
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "probability" in result
        assert result["prediction"] in [0, 1]

    def test_deterministic(self, simple_model):
        """同一输入多次预测结果一致."""
        model_path, _ = simple_model
        input_data = {
            "age": 35, "campaign": 1, "pdays": 999, "previous": 0,
            "emp_var_rate": -1.8, "cons_price_index": 96.0,
            "cons_conf_index": -40.0, "lending_rate3m": 4.0,
            "nr_employed": 5000.0,
            "job": "technician", "marital": "single", "education": "university.degree",
            "default": "no", "housing": "no", "loan": "yes",
            "contact": "telephone", "month": "jun", "day_of_week": "tue",
            "poutcome": "success",
        }
        r1 = predict_single(model_path, input_data)
        r2 = predict_single(model_path, input_data)
        assert r1["prediction"] == r2["prediction"]
        assert r1["probability"] == pytest.approx(r2["probability"])

    def test_model_not_found(self):
        """模型文件不存在抛出 FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="模型文件不存在"):
            predict_single("nonexistent/model.joblib", {"age": 30})
