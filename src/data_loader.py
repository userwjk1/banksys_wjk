"""数据加载与预处理模块.

负责银行营销 CSV 数据的加载、缺失值填充、分类特征编码和目标列转换。
训练时自动排除 `duration` 列以避免数据泄漏。
"""

import logging
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# 银行营销数据中的所有分类列
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]

# 训练/预测时必须排除的列(事后信息,导致数据泄漏)
LEAKAGE_COLS = ["duration"]

# 非特征列(建模时排除,保留 id 列用于 test.csv 输出)
META_COLS = ["id"]


def load_data(file_path: str) -> pd.DataFrame:
    """从 CSV 文件加载银行营销数据.

    Args:
        file_path: CSV 文件路径.

    Returns:
        原始 DataFrame.

    Raises:
        FileNotFoundError: 文件不存在时抛出.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"数据文件不存在: {file_path}\n"
            f"请确认文件路径正确,或将数据文件放入 data/ 目录。"
        )

    df = pd.read_csv(file_path)
    logger.info("已加载数据: %s, 形状 %s", file_path, df.shape)
    return df


def preprocess(
    df: pd.DataFrame,
    target_col: str | None = None,
    exclude_duration: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """数据预处理:缺失值填充 + 分类特征编码 + 目标列转换.

    Args:
        df: 原始 DataFrame.
        target_col: 目标列名,None 表示无目标列(预测模式).
        exclude_duration: 是否排除 duration 列以避免数据泄漏(默认 True).

    Returns:
        (特征 DataFrame, 目标 Series 或 None).
    """
    df = df.copy()

    # 分离目标列
    target = None
    if target_col and target_col in df.columns:
        target = df[target_col].map({"yes": 1, "no": 0}).copy()
        if target.isna().any():
            logger.warning("目标列存在无法映射的值(yes/no 之外),已转为 NaN")
        df = df.drop(columns=[target_col])

    # 排除非特征列与泄漏列
    drop_cols = [c for c in META_COLS + LEAKAGE_COLS if c in df.columns]
    if not exclude_duration and "duration" in drop_cols:
        drop_cols.remove("duration")
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 缺失值填充:数值列填中位数,分类列填众数
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug("数值列 %s 缺失值已填中位数 %.2f", col, median_val)
            else:
                mode_val = df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "unknown"
                df[col] = df[col].fillna(fill_val)
                logger.debug("分类列 %s 缺失值已填众数 %s", col, fill_val)

    # 分类特征编码
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    logger.info("预处理完成: 特征数=%d, 样本数=%d", df.shape[1], df.shape[0])
    return df, target


def get_processed_data(
    file_path: str,
    target_col: str | None = "subscribe",
) -> tuple[pd.DataFrame, pd.Series | None]:
    """加载并预处理数据的便捷函数.

    Args:
        file_path: CSV 文件路径.
        target_col: 目标列名,None 表示无目标列.

    Returns:
        (特征矩阵 X, 目标向量 y 或 None).
    """
    df = load_data(file_path)
    return preprocess(df, target_col=target_col, exclude_duration=True)
