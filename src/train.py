"""模型离线训练模块.

基于 scikit-learn Pipeline 训练二分类模型(LogisticRegression / RandomForest),
自动选择 AUC 更优的模型保存到 models/model.joblib。

用法:
    python -m src.train --data-path data/train.csv --model-path models/model.joblib
"""

import os
import argparse
import logging

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.data_loader import get_processed_data

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# 分类特征与数值特征(与 data_loader.CATEGORICAL_COLS 保持一致)
CAT_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]

MODEL_FACTORY = {
    "lr": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "rf": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
}


def build_pipeline(model_type: str = "rf") -> Pipeline:
    """构建包含预处理与分类器的 sklearn Pipeline.

    Args:
        model_type: "lr"(逻辑回归) 或 "rf"(随机森林).

    Returns:
        sklearn Pipeline.

    Raises:
        ValueError: 无效的 model_type.
    """
    if model_type not in MODEL_FACTORY:
        raise ValueError(
            f"无效的模型类型: {model_type!r},可选: {list(MODEL_FACTORY)}"
        )

    # 数值特征处理:填充缺失值 + 标准化
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 分类特征处理:填充缺失值 + OneHot 编码
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, None),  # None = 自动识别数值列
            ("cat", categorical_transformer, CAT_FEATURES),
        ],
        remainder="drop",
    )

    model = MODEL_FACTORY[model_type]
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "rf",
) -> tuple[Pipeline, dict]:
    """训练模型并返回评估指标.

    Args:
        X: 特征矩阵.
        y: 目标向量(0/1).
        model_type: 模型类型.

    Returns:
        (训练好的 Pipeline, 指标字典).
    """
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    pipe = build_pipeline(model_type)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
    }

    # 交叉验证 AUC
    cv_scores = cross_val_score(
        pipe, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
    )
    metrics["cv_auc_mean"] = float(cv_scores.mean())
    metrics["cv_auc_std"] = float(cv_scores.std())

    logger.info(
        "%s 评估: AUC=%.4f, Acc=%.4f, F1=%.4f",
        model_type.upper(), metrics["auc"], metrics["accuracy"], metrics["f1"],
    )

    return pipe, metrics


def save_model(model: Pipeline, path: str) -> None:
    """保存模型到磁盘.

    Args:
        model: 训练好的 sklearn Pipeline.
        path: 保存路径.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    logger.info("模型已保存: %s", path)


def load_model(path: str) -> Pipeline:
    """从磁盘加载模型.

    Args:
        path: 模型文件路径.

    Returns:
        sklearn Pipeline.

    Raises:
        FileNotFoundError: 模型文件不存在.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"模型文件不存在: {path}\n"
            f"请先运行训练脚本: python -m src.train --data-path data/train.csv"
        )
    return joblib.load(path)


def run(data_path: str, model_path: str) -> None:
    """执行完整训练流程:加载数据 → 训练多模型 → 选最优 → 保存.

    Args:
        data_path: 训练数据 CSV 路径.
        model_path: 模型保存路径.
    """
    print("=" * 60)
    print("  银行营销认购预测 — 模型离线训练")
    print("=" * 60)

    # 加载数据
    print(f"\n[1/4] 加载数据: {data_path}")
    X, y = get_processed_data(data_path, target_col="subscribe")
    print(f"  特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
    print(f"  认购率: {y.mean():.1%}")

    # 确认无泄漏
    assert "duration" not in X.columns, "数据泄漏: duration 列未被排除!"
    assert "subscribe" not in X.columns, "数据泄漏: 目标列未排除!"
    print("  ✓ 数据泄漏检查通过")

    # 训练多个模型
    print("\n[2/4] 训练模型...")
    results = {}
    for model_type in ["lr", "rf"]:
        _, metrics = train_and_evaluate(X, y, model_type=model_type)
        results[model_type] = metrics
        print(
            f"  {model_type.upper():>4s}: "
            f"AUC={metrics['auc']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}, "
            f"CV_AUC={metrics['cv_auc_mean']:.4f}±{metrics['cv_auc_std']:.4f}"
        )

    # 选最优
    print("\n[3/4] 选择最优模型...")
    best_type = max(results, key=lambda k: results[k]["auc"])
    best_metrics = results[best_type]
    print(f"  最优模型: {best_type.upper()} (AUC={best_metrics['auc']:.4f})")

    # 质量门槛
    assert best_metrics["auc"] >= 0.75, \
        f"模型 AUC {best_metrics['auc']:.3f} 未达标(>=0.75)"
    assert best_metrics["accuracy"] >= 0.80, \
        f"模型准确率 {best_metrics['accuracy']:.3f} 未达标(>=0.80)"
    print("  ✓ 模型质量达标")

    # 全量重训最优模型并保存
    best_pipe = build_pipeline(best_type)
    best_pipe.fit(X, y)

    print(f"\n[4/4] 保存模型: {model_path}")
    save_model(best_pipe, model_path)
    print("=" * 60)
    print("  ✓ 训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="银行营销认购预测模型训练")
    parser.add_argument(
        "--data-path", default="data/train.csv", help="训练数据 CSV 路径"
    )
    parser.add_argument(
        "--model-path", default="models/model.joblib", help="模型保存路径"
    )
    args = parser.parse_args()
    run(args.data_path, args.model_path)
