"""
generate_sample.py - 示例数据生成器

运行此脚本可生成一份符合格式要求的示例 Excel 成绩文件，
用于功能测试和演示。
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path

# ─── 配置 ─────────────────────────────────────────────────────
NUM_STUDENTS = 40       # 学生人数
SUBJECTS     = ["语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治"]
OUTPUT_FILE  = "示例成绩表.xlsx"

random.seed(42)
np.random.seed(42)


def generate_score(mean: float, std: float) -> float:
    """
    生成一个符合正态分布、范围 [0,100] 的分数（保留1位小数）。

    Args:
        mean (float): 均值
        std (float): 标准差

    Returns:
        float: 生成的分数
    """
    score = np.random.normal(mean, std)
    return round(float(np.clip(score, 0, 100)), 1)


def main():
    """生成示例数据并保存为 Excel 文件。"""
    # 学生基本信息
    rows = []
    for i in range(1, NUM_STUDENTS + 1):
        student_id = f"2024{i:03d}"
        # 姓氏 + 名字组合（仅用于演示）
        surnames = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨"
        names    = "明华伟志强勇敏静芳艳秀"
        name = random.choice(surnames) + random.choice(names)

        # 为每位学生随机生成"偏科程度"
        base = random.uniform(55, 88)   # 每人综合水平基线
        row = {"学号": student_id, "姓名": name}
        for subj in SUBJECTS:
            offset = random.uniform(-12, 12)
            row[subj] = generate_score(base + offset, 8)
        rows.append(row)

    df = pd.DataFrame(rows)

    # 随机植入少量缺失值（模拟缺考）
    for _ in range(3):
        r = random.randint(0, NUM_STUDENTS - 1)
        c = random.choice(SUBJECTS)
        df.at[r, c] = None

    # 保存
    output_path = Path(__file__).parent / OUTPUT_FILE
    df.to_excel(output_path, index=False)
    print(f"✅ 示例数据已生成：{output_path}")
    print(f"   学生数：{NUM_STUDENTS}，科目：{SUBJECTS}")


if __name__ == "__main__":
    main()
