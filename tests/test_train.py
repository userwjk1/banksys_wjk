"""测试 src/train.py — 模型离线训练."""

import os
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.data_loader import get_processed_data
from src.train import build_pipeline, train_and_evaluate, save_model, load_model


@pytest.fixture
def sample_data_path(tmp_path):
    """构建含目标列 subscribe 的模拟训练数据."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age": np.random.randint(20, 70, n),
        "job": np.random.choice(["admin.", "blue-collar", "technician", "services", "management"], n),
        "marital": np.random.choice(["married", "single", "divorced"], n),
        "education": np.random.choice(["high.school", "basic.9y", "professional.course", "university.degree"], n),
        "default": np.random.choice(["no", "yes"], n, p=[0.8, 0.2]),
        "housing": np.random.choice(["yes", "no"], n),
        "loan": np.random.choice(["no", "yes"], n),
        "contact": np.random.choice(["cellular", "telephone"], n),
        "month": np.random.choice(["may", "jun", "jul", "aug"], n),
        "day_of_week": np.random.choice(["mon", "tue", "wed", "thu", "fri"], n),
        "duration": np.random.randint(0, 500, n),
        "campaign": np.random.randint(1, 10, n),
        "pdays": np.random.choice([999, 0, 3, 6], n),
        "previous": np.random.randint(0, 5, n),
        "poutcome": np.random.choice(["nonexistent", "failure", "success"], n),
        "emp_var_rate": np.random.uniform(-3, 2, n),
        "cons_price_index": np.random.uniform(89, 97, n),
        "cons_conf_index": np.random.uniform(-45, -30, n),
        "lending_rate3m": np.random.uniform(0.5, 5.5, n),
        "nr_employed": np.random.uniform(4900, 5300, n),
        "subscribe": np.random.choice(["no", "yes"], n, p=[0.6, 0.4]),
    })
    path = tmp_path / "sample_train.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestBuildPipeline:
    """测试 build_pipeline."""

    def test_returns_sklearn_pipeline(self):
        """返回 sklearn Pipeline 对象."""
        pipe = build_pipeline(model_type="lr")
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_steps(self):
        """Pipeline 包含预处理和模型步骤."""
        pipe = build_pipeline(model_type="rf")
        assert len(pipe.steps) >= 2

    def test_invalid_model_raises(self):
        """无效模型类型抛出 ValueError."""
        with pytest.raises(ValueError):
            build_pipeline(model_type="invalid")


class TestTrainAndEvaluate:
    """测试 train_and_evaluate."""

    def test_returns_metrics(self, sample_data_path):
        """训练返回评估指标字典."""
        X, y = get_processed_data(sample_data_path, target_col="subscribe")
        pipe, metrics = train_and_evaluate(X, y, model_type="lr")
        assert isinstance(metrics, dict)
        for key in ("auc", "accuracy", "precision", "recall", "f1"):
            assert key in metrics, f"缺少指标: {key}"
            assert 0.0 <= metrics[key] <= 1.0

    def test_different_models(self, sample_data_path):
        """两种模型都能训练."""
        X, y = get_processed_data(sample_data_path, target_col="subscribe")
        _, metrics_lr = train_and_evaluate(X, y, model_type="lr")
        _, metrics_rf = train_and_evaluate(X, y, model_type="rf")
        assert metrics_lr["auc"] > 0
        assert metrics_rf["auc"] > 0


class TestSaveAndLoadModel:
    """测试模型保存与加载."""

    def test_save_and_load(self, sample_data_path, tmp_path):
        """保存后加载,模型类型一致."""
        X, y = get_processed_data(sample_data_path, target_col="subscribe")
        pipe, _ = train_and_evaluate(X, y, model_type="lr")

        model_path = str(tmp_path / "model.joblib")
        save_model(pipe, model_path)
        assert os.path.isfile(model_path)

        loaded = load_model(model_path)
        assert isinstance(loaded, Pipeline)

    def test_load_nonexistent_file(self):
        """加载不存在的模型抛出 FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="模型文件不存在"):
            load_model("nonexistent/model.joblib")
