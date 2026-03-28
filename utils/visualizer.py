"""
visualizer.py - 图表生成模块

使用 Plotly 生成交互式图表，使用 Matplotlib 生成可嵌入 PDF 的静态图表。
"""

import io
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")   # 非交互后端，适合服务端渲染
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ─── 常量定义 ───────────────────────────────────────────────
PRIMARY_COLOR = "#2E86AB"           # 主色调（深蓝）
ACCENT_COLOR  = "#A23B72"           # 强调色
SUCCESS_COLOR = "#2EC4B6"           # 成功/正向色
WARN_COLOR    = "#E76F51"           # 警告色
BG_COLOR      = "#F8F9FA"           # 背景色
GRID_COLOR    = "#E0E0E0"           # 网格线颜色

# 分数段颜色映射（对应 SCORE_LABELS 的六个区间）
DIST_COLORS = [
    "#E63946",  # 0-40  不及格（红）
    "#F4A261",  # 40-60 较差（橙）
    "#E9C46A",  # 60-70 中等（黄）
    "#2EC4B6",  # 70-80 良好（青）
    "#2E86AB",  # 80-90 优秀（蓝）
    "#1B4332",  # 90-100 卓越（深绿）
]

# Matplotlib 中文字体配置
rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 一、Plotly 交互式图表
# ══════════════════════════════════════════════════════════════

def plot_score_distribution(dist: pd.Series, subject: str) -> go.Figure:
    """
    生成单科成绩分段分布的 Plotly 交互柱状图。

    Args:
        dist (pd.Series): compute_score_distribution 返回的分布数据
        subject (str): 科目名称，用于标题

    Returns:
        go.Figure: Plotly 图表对象
    """
    labels = dist.index.tolist()
    values = dist.values.tolist()
    total = sum(values)

    # 计算各区间占比，用于悬停提示
    percentages = [f"{v/total*100:.1f}%" if total > 0 else "0%" for v in values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=DIST_COLORS,
            marker_line_color="white",
            marker_line_width=1.5,
            text=values,
            textposition="outside",
            customdata=percentages,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "人数：%{y}<br>"
                "占比：%{customdata}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(text=f"「{subject}」成绩分段分布", font=dict(size=16, color="#000000")),
        xaxis=dict(title="分数段", tickfont=dict(size=12)),
        yaxis=dict(title="人数", gridcolor=GRID_COLOR, tickfont=dict(size=12)),
        plot_bgcolor="white",
        paper_bgcolor=BG_COLOR,
        margin=dict(t=60, b=40, l=50, r=20),
        showlegend=False,
    )
    return fig


def plot_all_distributions(all_dist: dict[str, pd.Series]) -> go.Figure:
    """
    生成所有科目分数段分布的多子图 Plotly 图表。

    Args:
        all_dist (dict[str, pd.Series]): compute_all_distributions 的返回值

    Returns:
        go.Figure: 包含多个子图的 Plotly 图表
    """
    subjects = list(all_dist.keys())
    n = len(subjects)
    # 动态计算子图行列数：尽量保持接近正方形
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f"「{s}」" for s in subjects],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for idx, subject in enumerate(subjects):
        row = idx // ncols + 1
        col = idx % ncols + 1
        dist = all_dist[subject]
        values = dist.values.tolist()
        total = sum(values)

        fig.add_trace(
            go.Bar(
                x=dist.index.tolist(),
                y=values,
                marker_color=DIST_COLORS,
                marker_line_color="white",
                marker_line_width=1,
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>人数：%{y}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    height = max(350, nrows * 280)
    fig.update_layout(
        title=dict(text="各科成绩分段分布总览", font=dict(size=18, color="#000000")),
        plot_bgcolor="white",
        paper_bgcolor=BG_COLOR,
        height=height,
        margin=dict(t=80, b=40),
    )
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def plot_radar_chart(student_row: pd.Series, score_cols: list[str]) -> go.Figure:
    """
    生成单个学生各科成绩的 Plotly 雷达图。

    Args:
        student_row (pd.Series): 单行学生数据（含姓名和各科成绩）
        score_cols (list[str]): 成绩列名列表

    Returns:
        go.Figure: Plotly 雷达图对象
    """
    name = student_row.get("姓名", "未知")
    scores = [float(student_row.get(col, 0) or 0) for col in score_cols]

    # 闭合多边形：首尾相接
    categories = score_cols + [score_cols[0]]
    values = scores + [scores[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor=f"rgba(46, 134, 171, 0.25)",
            line=dict(color=PRIMARY_COLOR, width=2),
            marker=dict(color=PRIMARY_COLOR, size=6),
            name=name,
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
                gridcolor=GRID_COLOR,
            ),
            angularaxis=dict(tickfont=dict(size=12)),
            bgcolor="white",
        ),
        title=dict(
            text=f"{name} — 各科成绩雷达图",
            font=dict(size=15, color="#000000"),
        ),
        paper_bgcolor=BG_COLOR,
        showlegend=False,
        margin=dict(t=60, b=30, l=60, r=60),
    )
    return fig


def plot_class_avg_bar(stats_df: pd.DataFrame) -> go.Figure:
    """
    生成各科班级平均分对比柱状图。

    Args:
        stats_df (pd.DataFrame): compute_class_stats 返回的统计表

    Returns:
        go.Figure: Plotly 柱状图
    """
    subjects = stats_df.columns.tolist()
    avgs = stats_df.loc["平均分"].values.tolist()
    maxs = stats_df.loc["最高分"].values.tolist()
    mins = stats_df.loc["最低分"].values.tolist()

    fig = go.Figure()

    # 平均分主柱
    fig.add_trace(go.Bar(
        name="平均分",
        x=subjects,
        y=avgs,
        marker_color=PRIMARY_COLOR,
        text=[f"{v:.1f}" for v in avgs],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>平均分：%{y:.1f}<extra></extra>",
    ))

    # 最高分散点
    fig.add_trace(go.Scatter(
        name="最高分",
        x=subjects,
        y=maxs,
        mode="markers",
        marker=dict(color=SUCCESS_COLOR, size=10, symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>最高分：%{y}<extra></extra>",
    ))

    # 最低分散点
    fig.add_trace(go.Scatter(
        name="最低分",
        x=subjects,
        y=mins,
        mode="markers",
        marker=dict(color=WARN_COLOR, size=10, symbol="cross"),
        hovertemplate="<b>%{x}</b><br>最低分：%{y}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="各科成绩统计对比", font=dict(size=16, color="#000000")),
        xaxis=dict(title="科目", tickfont=dict(size=12)),
        yaxis=dict(title="分数", range=[0, 110], gridcolor=GRID_COLOR),
        plot_bgcolor="white",
        paper_bgcolor=BG_COLOR,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=50, r=20),
    )
    return fig


def plot_score_heatmap(df: pd.DataFrame, score_cols: list[str]) -> go.Figure:
    """
    生成全班学生各科成绩热力图，直观展示整体成绩格局。

    Args:
        df (pd.DataFrame): 经过验证的 DataFrame
        score_cols (list[str]): 成绩列名列表

    Returns:
        go.Figure: Plotly 热力图
    """
    # 取前50名以避免图表过密
    display_df = df.head(50)
    z = display_df[score_cols].values
    y_labels = (display_df["姓名"] + "(" + display_df["学号"] + ")").tolist()

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=score_cols,
        y=y_labels,
        colorscale=[
            [0,   "#E63946"],
            [0.4, "#F4A261"],
            [0.6, "#E9C46A"],
            [0.7, "#2EC4B6"],
            [0.85,"#2E86AB"],
            [1,   "#1B4332"],
        ],
        zmin=0,
        zmax=100,
        hoverongaps=False,
        hovertemplate="学生：%{y}<br>科目：%{x}<br>分数：%{z}<extra></extra>",
        colorbar=dict(title="分数"),
    ))

    height = max(400, len(y_labels) * 22)
    fig.update_layout(
        title=dict(text="全班成绩热力图", font=dict(size=16, color="#000000")),
        xaxis=dict(side="top", tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        paper_bgcolor=BG_COLOR,
        margin=dict(t=80, b=20, l=160, r=60),
        height=min(height, 900),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# 二、Matplotlib 静态图表（用于 PDF 嵌入）
# ══════════════════════════════════════════════════════════════

def mpl_distribution_bar(dist: pd.Series, subject: str) -> bytes:
    """
    使用 Matplotlib 生成单科分段柱状图并返回 PNG 字节流。

    Args:
        dist (pd.Series): 分布数据（索引为区间标签，值为人数）
        subject (str): 科目名称

    Returns:
        bytes: PNG 图片字节流，可直接写入 PDF
    """
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="white")

    labels = dist.index.tolist()
    values = dist.values.tolist()
    total = max(sum(values), 1)

    bars = ax.bar(labels, values, color=DIST_COLORS, edgecolor="white", linewidth=1.2)

    # 在柱顶标注人数和百分比
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val}\n({val/total*100:.0f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(f"「{subject}」成绩分段分布", fontsize=13, pad=10)
    ax.set_xlabel("分数段", fontsize=10)
    ax.set_ylabel("人数", fontsize=10)
    ax.set_ylim(0, max(values) * 1.3 + 1)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def mpl_radar_chart(student_row: pd.Series, score_cols: list[str]) -> bytes:
    """
    使用 Matplotlib 生成学生雷达图并返回 PNG 字节流。

    Args:
        student_row (pd.Series): 单行学生数据
        score_cols (list[str]): 成绩列名列表

    Returns:
        bytes: PNG 图片字节流
    """
    name = student_row.get("姓名", "未知")
    scores = [float(student_row.get(col, 0) or 0) for col in score_cols]
    n = len(score_cols)

    # 计算角度：均匀分布，闭合
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    scores_closed = scores + scores[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True), facecolor="white")

    ax.plot(angles, scores_closed, color=PRIMARY_COLOR, linewidth=2)
    ax.fill(angles, scores_closed, color=PRIMARY_COLOR, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(score_cols, fontsize=9)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_ylim(0, 100)
    ax.set_title(f"{name}", fontsize=12, pad=15)
    ax.grid(color=GRID_COLOR)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def mpl_avg_bar(stats_df: pd.DataFrame) -> bytes:
    """
    使用 Matplotlib 生成各科平均分柱状图并返回 PNG 字节流。

    Args:
        stats_df (pd.DataFrame): 统计表

    Returns:
        bytes: PNG 图片字节流
    """
    subjects = stats_df.columns.tolist()
    avgs = stats_df.loc["平均分"].values.tolist()

    # 生成渐变色：根据平均分高低映射颜色
    colors = [
        DIST_COLORS[0] if (v or 0) < 60 else
        DIST_COLORS[2] if (v or 0) < 75 else
        DIST_COLORS[3] if (v or 0) < 85 else
        DIST_COLORS[4]
        for v in avgs
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(subjects) * 1.2), 4), facecolor="white")
    bars = ax.bar(subjects, avgs, color=colors, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, avgs):
        if val is not None and not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_title("各科班级平均分", fontsize=13, pad=10)
    ax.set_ylabel("平均分", fontsize=10)
    ax.set_ylim(0, 110)
    ax.axhline(y=60, color="#E63946", linestyle="--", linewidth=1, alpha=0.7, label="及格线")
    ax.axhline(y=85, color="#2EC4B6", linestyle="--", linewidth=1, alpha=0.7, label="优秀线")
    ax.legend(fontsize=8)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
