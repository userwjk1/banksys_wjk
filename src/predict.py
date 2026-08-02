"""模型加载与在线预测模块.

支持从磁盘加载训练好的 Pipeline 并对单条客户特征进行预测,
包含输入校验以防止非法值导致模型异常。
"""

import logging
import pandas as pd
from src.train import load_model

logger = logging.getLogger(__name__)

# 各数值特征的合理范围(用于前端校验)
NUMERIC_RANGES = {
    "age": (17, 100),
    "campaign": (1, 50),
    "pdays": (0, 999),
    "previous": (0, 50),
    "emp_var_rate": (-5.0, 5.0),
    "cons_price_index": (85.0, 100.0),
    "cons_conf_index": (-55.0, -20.0),
    "lending_rate3m": (0.0, 10.0),
    "nr_employed": (4500.0, 5500.0),
}


def validate_input(
    data: dict,
    feature_names: list[str],
) -> tuple[bool, list[str]]:
    """校验输入值的合理性.

    Args:
        data: 用户输入的键值对.
        feature_names: 模型期望的特征名列表.

    Returns:
        (是否通过, 错误消息列表).
    """
    errors = []

    # 检查必填字段
    for feat in feature_names:
        if feat not in data or data[feat] is None or data[feat] == "":
            errors.append(f"缺少必填字段: {feat}")

    if errors:
        return False, errors

    # 数值范围校验
    for feat, (lo, hi) in NUMERIC_RANGES.items():
        if feat in data:
            try:
                val = float(data[feat])
                if val < lo or val > hi:
                    errors.append(f"{feat} 超出合理范围 [{lo}, {hi}],当前值: {val}")
            except (ValueError, TypeError):
                errors.append(f"{feat} 不是有效数字: {data[feat]}")

    return len(errors) == 0, errors


def predict_single(model_path: str, input_data: dict) -> dict:
    """对单条客户特征进行认购预测.

    Args:
        model_path: 训练好的模型文件路径.
        input_data: 客户特征字典,键为列名,值为特征值.

    Returns:
        {"prediction": 0|1, "probability": float, "label": "会认购"|"不会认购"}

    Raises:
        FileNotFoundError: 模型文件不存在.
    """
    model = load_model(model_path)

    # 构造 DataFrame(与训练时列顺序保持一致)
    try:
        feature_names = model.feature_names_in_
    except AttributeError:
        # 如果是 ColumnTransformer+Pipeline,从 preprocessor 获取
        feature_names = list(input_data.keys())

    df = pd.DataFrame([input_data])

    # 确保所有必要列存在且顺序正确
    for col in feature_names:
        if col not in df.columns:
            df[col] = "unknown"

    df = df[feature_names]

    prob = float(model.predict_proba(df)[0, 1])
    pred = 1 if prob >= 0.5 else 0

    result = {
        "prediction": pred,
        "probability": round(prob, 4),
        "label": "会认购" if pred == 1 else "不会认购",
    }
    logger.info("预测完成: prob=%.4f, pred=%d", prob, pred)
    return result
