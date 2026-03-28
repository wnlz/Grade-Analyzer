"""
data_processor.py - 数据处理模块

负责 Excel 文件的读取、格式验证、数据清洗，
以及班级统计、分段分布等核心计算逻辑。
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

# ─── 常量定义 ───────────────────────────────────────────────
DEFAULT_MIN_SCORE = 0     # 默认最低合法分数
DEFAULT_MAX_SCORE = 100    # 默认最高合法分数
MAX_FILE_SIZE_MB = 10   # 文件大小限制（MB）
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# 科目分数范围配置（可根据需要扩展）
SUBJECT_SCORE_RANGES: Dict[str, Tuple[int, int]] = {
    "语文": (0, 150),
    "数学": (0, 150),
    "英语": (0, 150),
    # 其他科目使用默认范围 0-100
}

# 分数段区间定义：[左闭, 右开) → 显示标签
# 100分制的分数段区间
DEFAULT_SCORE_BINS = [0, 40, 60, 70, 80, 90, 100]
DEFAULT_SCORE_LABELS = ["0-40", "40-60", "60-70", "70-80", "80-90", "90-100"]
# 150分制的分数段区间
SCORE_BINS_150 = [0, 40, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
SCORE_LABELS_150 = ["0-40", "40-60", "60-70", "70-80", "80-90", "90-100", "100-110", "110-120", "120-130", "130-140", "140-150"]

# 列索引约定
COL_ID = 0      # 学号列索引
COL_NAME = 1    # 姓名列索引
COL_SCORE_START = 2  # 成绩列起始索引

logger = logging.getLogger(__name__)

def get_subject_score_range(subject: str) -> Tuple[int, int]:
    """
    获取科目的分数范围。
    
    Args:
        subject: 科目名称
        
    Returns:
        Tuple[int, int]: (min_score, max_score)
    """
    if subject in SUBJECT_SCORE_RANGES:
        return SUBJECT_SCORE_RANGES[subject]
    return (DEFAULT_MIN_SCORE, DEFAULT_MAX_SCORE)


def get_score_bins_and_labels(subject: str) -> Tuple[list, list]:
    """
    根据科目获取对应的分数段区间和标签。
    
    Args:
        subject: 科目名称
        
    Returns:
        Tuple[list, list]: (bins, labels)
    """
    min_score, max_score = get_subject_score_range(subject)
    
    # 如果是150分制，使用150分制的分段
    if max_score == 150:
        return SCORE_BINS_150, SCORE_LABELS_150
    
    # 默认使用100分制的分段
    return DEFAULT_SCORE_BINS, DEFAULT_SCORE_LABELS

# ══════════════════════════════════════════════════════════════
# 一、文件读取
# ══════════════════════════════════════════════════════════════

def read_excel(file_obj) -> tuple[pd.DataFrame, list[str]]:
    """
    读取上传的 Excel 文件，返回原始 DataFrame 及警告列表。

    支持 .xlsx 和 .xls 格式。
    第一列为学号，第二列为姓名，后续列为各科成绩。

    Args:
        file_obj: Streamlit UploadedFile 对象

    Returns:
        tuple:
            - pd.DataFrame: 读取到的原始数据
            - list[str]: 警告信息列表（非阻断性问题）

    Raises:
        ValueError: 文件格式不合法或列数不足时抛出
        IOError: 文件读取失败时抛出
    """
    warnings: list[str] = []

    # ── 文件大小校验 ──
    file_obj.seek(0, 2)  # 移动到文件末尾
    size = file_obj.tell()
    file_obj.seek(0)
    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"文件大小 {size / 1024 / 1024:.1f}MB 超过 {MAX_FILE_SIZE_MB}MB 限制"
        )

    # ── 读取 Excel ──
    try:
        df = pd.read_excel(file_obj, header=0, dtype=str)
    except Exception as e:
        logger.error(f"Excel 读取失败：{e}")
        raise IOError(f"无法读取 Excel 文件，请确认格式正确：{e}") from e

    # ── 基础列数校验 ──
    if df.shape[1] < 3:
        raise ValueError(
            "数据列数不足：至少需要「学号」「姓名」和至少一科成绩列（共 3 列以上）"
        )

    logger.info(f"成功读取 Excel：{df.shape[0]} 行，{df.shape[1]} 列")
    return df, warnings


# ══════════════════════════════════════════════════════════════
# 二、数据验证与清洗
# ══════════════════════════════════════════════════════════════

def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    对原始 DataFrame 进行数据验证和清洗。

    验证内容：
    1. 学号与姓名列不为空
    2. 成绩列可转换为数值
    3. 分数范围根据科目配置进行校验
    4. 缺失值处理（标注警告，保留行）

    Args:
        df (pd.DataFrame): 来自 read_excel 的原始数据

    Returns:
        tuple:
            - pd.DataFrame: 清洗后的 DataFrame（成绩列为 float，可含 NaN）
            - list[str]: 错误信息列表（阻断性问题，需用户修正）
            - list[str]: 警告信息列表（非阻断性，系统自动处理）
    """
    errors: list[str] = []
    warnings: list[str] = []
    clean_df = df.copy()

    # ── 重命名固定列 ──
    col_names = list(clean_df.columns)
    col_names[COL_ID] = "学号"
    col_names[COL_NAME] = "姓名"
    clean_df.columns = col_names

    # ── 学号与姓名空值检测 ──
    id_null = clean_df["学号"].isna() | (clean_df["学号"].astype(str).str.strip() == "")
    name_null = clean_df["姓名"].isna() | (clean_df["姓名"].astype(str).str.strip() == "")
    if id_null.any():
        null_rows = (id_null[id_null].index + 2).tolist()  # +2 因为 header 占 1 行，索引从 0 开始
        warnings.append(f"第 {null_rows} 行学号为空，已跳过这些行")
        clean_df = clean_df[~id_null].copy()
    if name_null.any():
        null_rows = (name_null[name_null & ~id_null].index + 2).tolist()
        if null_rows:
            warnings.append(f"第 {null_rows} 行姓名为空，已跳过这些行")
        clean_df = clean_df[~(name_null & ~id_null)].copy()

    # ── 成绩列数值转换与校验 ──
    score_cols = list(clean_df.columns[COL_SCORE_START:])
    for col in score_cols:
        # 获取该科目的分数范围
        min_score, max_score = get_subject_score_range(col)
        
        # 将字符串转为数值，无法转换的变为 NaN
        numeric_col = pd.to_numeric(clean_df[col], errors="coerce")

        # 记录转换失败（非数字）的单元格
        non_numeric = clean_df[col].notna() & numeric_col.isna()
        if non_numeric.any():
            rows = (non_numeric[non_numeric].index + 2).tolist()
            warnings.append(f"科目「{col}」第 {rows} 行含非数字值，已置为空缺")

        # 检测超范围分数
        out_of_range = numeric_col.notna() & (
            (numeric_col < min_score) | (numeric_col > max_score)
        )
        if out_of_range.any():
            bad_vals = numeric_col[out_of_range].tolist()
            rows = (out_of_range[out_of_range].index + 2).tolist()
            errors.append(
                f"科目「{col}」第 {rows} 行分数 {bad_vals} 超出合法范围 "
                f"[{min_score}, {max_score}]"
            )

        clean_df[col] = numeric_col

    # ── 缺失值统计提示 ──
    missing_count = clean_df[score_cols].isna().sum().sum()
    if missing_count > 0:
        warnings.append(f"共有 {missing_count} 个成绩缺失，统计时将自动排除缺失值")

    # ── 规范化学号与姓名为字符串 ──
    clean_df["学号"] = clean_df["学号"].astype(str).str.strip()
    clean_df["姓名"] = clean_df["姓名"].astype(str).str.strip()
    clean_df = clean_df.reset_index(drop=True)

    logger.info(
        f"数据验证完成：{len(errors)} 个错误，{len(warnings)} 个警告，"
        f"有效行数 {len(clean_df)}"
    )
    return clean_df, errors, warnings

# ══════════════════════════════════════════════════════════════
# 三、统计分析
# ══════════════════════════════════════════════════════════════

def compute_class_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每科的班级整体统计指标。

    统计项：平均分、最高分、最低分、标准差、有效人数。

    Args:
        df (pd.DataFrame): 经过 validate_and_clean 的 DataFrame

    Returns:
        pd.DataFrame: 索引为统计指标名，列为各科目名的统计表
    """
    score_cols = list(df.columns[COL_SCORE_START:])
    stats = {}

    for col in score_cols:
        series = df[col].dropna()
        stats[col] = {
            "平均分": round(series.mean(), 2) if len(series) > 0 else np.nan,
            "最高分": series.max() if len(series) > 0 else np.nan,
            "最低分": series.min() if len(series) > 0 else np.nan,
            "标准差": round(series.std(ddof=1), 2) if len(series) > 1 else np.nan,
            "有效人数": len(series),
        }

    stats_df = pd.DataFrame(stats)
    logger.info("班级统计计算完成")
    return stats_df


def compute_score_distribution(series: pd.Series, subject: str = None) -> pd.Series:
    """
    计算单科成绩的分数段分布（各区间人数）。

    区间为左闭右开：[0,40), [40,60), [60,70), [70,80), [80,90), [90,100], ...
    注：最后区间 [90,100] 包含100分（通过 right=True + 末尾包含特殊处理）

    Args:
        series (pd.Series): 单科成绩列（含 NaN）
        subject (str): 科目名称，用于确定分数段区间

    Returns:
        pd.Series: 索引为区间标签，值为各区间人数
    """
    valid = series.dropna()

    if len(valid) == 0:
        # 如果没有有效数据，返回全零分布
        _, labels = get_score_bins_and_labels(subject or "")
        return pd.Series(0, index=labels)

    # 获取该科目的分数段区间
    bins, labels = get_score_bins_and_labels(subject or "")

    # pd.cut 使用 right=False 使区间为左闭右开
    dist = pd.cut(
        valid,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    ).value_counts().reindex(labels, fill_value=0)

    return dist


def compute_all_distributions(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    批量计算所有科目的分数段分布。

    Args:
        df (pd.DataFrame): 经过验证的 DataFrame

    Returns:
        dict[str, pd.Series]: key 为科目名，value 为分布 Series
    """
    score_cols = list(df.columns[COL_SCORE_START:])
    distributions = {}

    for col in score_cols:
        distributions[col] = compute_score_distribution(df[col], subject=col)

    return distributions


def compute_student_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每位学生的跨科目汇总信息。
    计算项：总分、平均分、最高分、最低分、各科成绩。

    注意：总分和平均分计算时需要考虑不同科目的满分可能不同。
    如果需要计算标准分或百分比，可以进一步扩展。

    Args:
        df (pd.DataFrame): 经过验证的 DataFrame

    Returns:
        pd.DataFrame: 在原 df 基础上追加汇总列的新 DataFrame
    """
    score_cols = list(df.columns[COL_SCORE_START:])
    result = df.copy()

    score_data = result[score_cols]
    result["总分"] = score_data.sum(axis=1, skipna=True).round(2)
    result["平均分"] = score_data.mean(axis=1, skipna=True).round(2)
    result["最高科目分"] = score_data.max(axis=1, skipna=True)
    result["最低科目分"] = score_data.min(axis=1, skipna=True)

    # 按总分降序排名
    result["排名"] = result["总分"].rank(ascending=False, method="min").astype(int)
    result = result.sort_values("排名").reset_index(drop=True)

    logger.info("学生汇总计算完成")
    return result


def get_score_columns(df: pd.DataFrame) -> list[str]:
    """
    提取 DataFrame 中的成绩列名列表。

    Args:
        df (pd.DataFrame): 经过 validate_and_clean 的 DataFrame

    Returns:
        list[str]: 成绩列名列表
    """
    return list(df.columns[COL_SCORE_START:])

def set_subject_score_range(subject: str, min_score: int, max_score: int) -> None:
    """
    动态设置科目的分数范围。
    
    Args:
        subject: 科目名称
        min_score: 最低分数
        max_score: 最高分数
    """
    SUBJECT_SCORE_RANGES[subject] = (min_score, max_score)
    logger.info(f"设置科目「{subject}」分数范围为 [{min_score}, {max_score}]")

def get_all_subject_ranges() -> Dict[str, Tuple[int, int]]:
    """
    获取所有科目的分数范围配置。
    
    Returns:
        Dict[str, Tuple[int, int]]: 科目分数范围字典
    """
    return SUBJECT_SCORE_RANGES.copy()

def delete_subject_score_range(subject_name: str) -> bool:
    """
    删除指定科目的分数范围配置。

    Args:
        subject_name: 科目名称

    Returns:
        bool: 是否删除成功
    """
    global SUBJECT_SCORE_RANGES
    if subject_name in SUBJECT_SCORE_RANGES:
        del SUBJECT_SCORE_RANGES[subject_name]
        logger.info(f"已删除科目范围配置: {subject_name}")
        return True
    logger.warning(f"尝试删除不存在的科目: {subject_name}")
    return False