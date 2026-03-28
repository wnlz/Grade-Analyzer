"""
pdf_generator.py - PDF 报告生成模块

使用 ReportLab 生成专业的成绩分析报告 PDF。
报告包含：封面、班级整体统计、分科分析、学生个人分析、教师评价。
"""

import io
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

# ─── 常量定义 ───────────────────────────────────────────────
PRIMARY_HEX   = "#2E86AB"
PRIMARY_COLOR = colors.HexColor(PRIMARY_HEX)
ACCENT_COLOR  = colors.HexColor("#A23B72")
LIGHT_BG      = colors.HexColor("#F8F9FA")
DARK_TEXT     = colors.HexColor("#000000")
GRAY_TEXT     = colors.HexColor("#555555")
BORDER_COLOR  = colors.HexColor("#DEE2E6")
SUCCESS_COLOR = colors.HexColor("#2EC4B6")
WARN_COLOR    = colors.HexColor("#E76F51")

PAGE_WIDTH, PAGE_HEIGHT = A4      # 595.27 x 841.89 points
MARGIN = 2.0 * cm                 # 页边距


def _register_fonts():
    """
    尝试注册中文字体，失败时回退到系统默认字体（不中断程序）。
    """
    font_candidates = [
        ("SimHei",   "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        ("SimHei",   "/System/Library/Fonts/PingFang.ttc"),
        ("SimHei",   "C:/Windows/Fonts/simhei.ttf"),
        ("SimHei",   "/usr/share/fonts/truetype/arphic/uming.ttc"),
    ]
    for font_name, path in font_candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, path))
                logger.info(f"注册字体成功：{font_name} <- {path}")
                return font_name
            except Exception as e:
                logger.warning(f"字体注册失败：{path} - {e}")
    logger.warning("未找到中文字体，PDF 中文可能显示为乱码")
    return "Helvetica"


FONT_NAME = _register_fonts()


def _style(
    name: str,
    font_size: int = 11,
    leading: int = 16,
    alignment=TA_LEFT,
    text_color=DARK_TEXT,
    bold: bool = False,
    space_before: float = 0,
    space_after: float = 0,
) -> ParagraphStyle:
    """
    快速创建 ReportLab ParagraphStyle 的辅助函数。

    Args:
        name (str): 样式唯一名称
        font_size (int): 字号
        leading (int): 行高
        alignment: 对齐方式
        text_color: 文字颜色
        bold (bool): 是否加粗（通过字体名后缀实现）
        space_before (float): 段前间距
        space_after (float): 段后间距

    Returns:
        ParagraphStyle: 配置好的段落样式
    """
    return ParagraphStyle(
        name=name,
        fontName=FONT_NAME,
        fontSize=font_size,
        leading=leading,
        alignment=alignment,
        textColor=text_color,
        spaceAfter=space_after,
        spaceBefore=space_before,
    )


# ══════════════════════════════════════════════════════════════
# 样式定义
# ══════════════════════════════════════════════════════════════
STYLES = {
    "title":      _style("title",      font_size=24, leading=32, alignment=TA_CENTER, bold=True),
    "subtitle":   _style("subtitle",   font_size=14, leading=20, alignment=TA_CENTER, text_color=GRAY_TEXT),
    "h1":         _style("h1",         font_size=16, leading=22, bold=True, space_before=10, space_after=6),
    "h2":         _style("h2",         font_size=13, leading=18, bold=True, space_before=8, space_after=4),
    "body":       _style("body",       font_size=10, leading=15, space_after=4),
    "body_c":     _style("body_c",     font_size=10, leading=15, alignment=TA_CENTER),
    "caption":    _style("caption",    font_size=8,  leading=12, text_color=GRAY_TEXT, alignment=TA_CENTER),
    "table_hdr":  _style("table_hdr",  font_size=9,  leading=13, alignment=TA_CENTER, text_color=colors.white, bold=True),
    "table_cell": _style("table_cell", font_size=9,  leading=13, alignment=TA_CENTER),
    "note":       _style("note",       font_size=9,  leading=13, text_color=GRAY_TEXT),
    "teacher":    _style("teacher",    font_size=10, leading=16, alignment=TA_JUSTIFY, space_after=6),
}


def _section_header(title: str) -> list:
    """
    生成带有装饰线的节标题 Flowable 列表。

    Args:
        title (str): 节标题文本

    Returns:
        list: 包含 Paragraph 和 HRFlowable 的列表
    """
    return [
        Spacer(1, 0.4 * cm),
        Paragraph(f"▌ {title}", STYLES["h1"]),
        HRFlowable(width="100%", thickness=1.5, color=PRIMARY_COLOR, spaceAfter=6),
    ]


def _stats_table(stats_df: pd.DataFrame) -> Table:
    """
    将统计 DataFrame 转换为 ReportLab Table（带样式）。

    Args:
        stats_df (pd.DataFrame): compute_class_stats 返回的统计表

    Returns:
        Table: 格式化后的 ReportLab 表格
    """
    # 构建表格数据：行=统计项，列=科目
    subjects = stats_df.columns.tolist()
    metrics  = stats_df.index.tolist()

    header_row = [Paragraph("指标", STYLES["table_hdr"])] + [
        Paragraph(s, STYLES["table_hdr"]) for s in subjects
    ]
    data = [header_row]

    for metric in metrics:
        row = [Paragraph(metric, STYLES["table_cell"])]
        for subj in subjects:
            val = stats_df.loc[metric, subj]
            cell_text = str(int(val)) if metric == "有效人数" else (
                f"{val:.2f}" if not (isinstance(val, float) and np.isnan(val)) else "-"
            )
            row.append(Paragraph(cell_text, STYLES["table_cell"]))
        data.append(row)

    col_width = (PAGE_WIDTH - 2 * MARGIN) / (len(subjects) + 1)
    table = Table(data, colWidths=[col_width] * (len(subjects) + 1), repeatRows=1)

    table.setStyle(TableStyle([
        # 表头
        ("BACKGROUND",  (0, 0), (-1, 0), PRIMARY_COLOR),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        # 隔行着色
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF6FB")]),
        # 通用
        ("FONTNAME",    (0, 0), (-1, -1), FONT_NAME),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",        (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _student_table(summary_df: pd.DataFrame, score_cols: list[str]) -> Table:
    """
    生成学生成绩汇总表。

    Args:
        summary_df (pd.DataFrame): compute_student_summary 返回的 DataFrame
        score_cols (list[str]): 成绩列名列表

    Returns:
        Table: ReportLab 表格
    """
    display_cols = ["排名", "学号", "姓名"] + score_cols + ["总分", "平均分"]
    sub_df = summary_df[display_cols].head(50)  # 最多展示50行

    # 表头
    header = [Paragraph(c, STYLES["table_hdr"]) for c in display_cols]
    data = [header]

    for _, row in sub_df.iterrows():
        tr = []
        for col in display_cols:
            val = row[col]
            if isinstance(val, float):
                text = f"{val:.1f}" if not np.isnan(val) else "-"
            else:
                text = str(val)
            tr.append(Paragraph(text, STYLES["table_cell"]))
        data.append(tr)

    total_width = PAGE_WIDTH - 2 * MARGIN
    fixed_cols  = 3   # 排名+学号+姓名
    score_n     = len(score_cols)
    summary_n   = 2   # 总分+平均分

    fixed_w  = total_width * 0.28 / fixed_cols
    score_w  = total_width * 0.52 / max(score_n, 1)
    summary_w= total_width * 0.20 / summary_n
    col_widths = [fixed_w] * fixed_cols + [score_w] * score_n + [summary_w] * summary_n

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), PRIMARY_COLOR),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF6FB")]),
        ("FONTNAME",      (0, 0), (-1, -1), FONT_NAME),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",          (0, 0), (-1, -1), 0.4, BORDER_COLOR),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


# ══════════════════════════════════════════════════════════════
# 主入口：生成 PDF
# ══════════════════════════════════════════════════════════════

def generate_report(
    stats_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    all_dist: dict,
    score_cols: list[str],
    chart_images: dict[str, bytes],
    teacher_comment: str = "",
    school_name: str = "学校名称",
    class_name:  str = "班级",
    logo_path:   Optional[str] = None,
    selected_outputs: Optional[dict] = None,
) -> bytes:
    """
    生成完整的成绩分析报告 PDF 并返回字节流。

    Args:
        stats_df (pd.DataFrame): 班级统计表（来自 compute_class_stats）
        summary_df (pd.DataFrame): 学生汇总表（来自 compute_student_summary）
        all_dist (dict): 各科分布数据（来自 compute_all_distributions）
        score_cols (list[str]): 成绩列名列表
        chart_images (dict[str, bytes]): 图表名称 -> PNG 字节流，键包括：
            "avg_bar"（平均分柱图）、"dist_{科目名}"（分布图）、
            "radar_{学号}"（雷达图）
        teacher_comment (str): 教师评语文本
        school_name (str): 学校名称，显示在封面
        class_name (str): 班级名称，显示在封面
        logo_path (Optional[str]): 学校 logo 图片路径，None 时不显示

    Returns:
        bytes: PDF 文件字节流
    """
    # 若未提供 selected_outputs，则默认全部生成（保持向后兼容）
    if selected_outputs is None:
        selected_outputs = {
            "class_stats": True,
            "dist_chart": True,
            "student_table": True,
            "radar_chart": True,
            "heatmap": False,   # PDF 暂不支持热力图，设为 False
        }
    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )

    # ── 页面模板（含页眉页脚） ──
    def _header_footer(canvas, document):
        canvas.saveState()
        # 页眉
        canvas.setFillColor(PRIMARY_COLOR)
        canvas.rect(0, PAGE_HEIGHT - 0.8 * cm, PAGE_WIDTH, 0.8 * cm, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont(FONT_NAME, 9)
        canvas.drawString(MARGIN, PAGE_HEIGHT - 0.58 * cm, f"{school_name}  {class_name}  成绩分析报告")
        canvas.drawRightString(
            PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 0.58 * cm,
            datetime.now().strftime("%Y-%m-%d")
        )
        # 页脚
        canvas.setFillColor(GRAY_TEXT)
        canvas.setFont(FONT_NAME, 8)
        canvas.drawCentredString(PAGE_WIDTH / 2, 0.5 * cm, f"- {document.page} -")
        canvas.restoreState()

    frame = Frame(MARGIN, MARGIN, PAGE_WIDTH - 2 * MARGIN, PAGE_HEIGHT - 2 * MARGIN - 0.8 * cm)
    template = PageTemplate(id="main", frames=[frame], onPage=_header_footer)
    doc.addPageTemplates([template])

    story = []

    # ════════════════════════════════════════════════════
    # 封面
    # ════════════════════════════════════════════════════
    story.append(Spacer(1, 3 * cm))

    # Logo（可选）
    if logo_path and os.path.exists(logo_path):
        try:
            img = Image(logo_path, width=4 * cm, height=4 * cm, kind="proportional")
            img.hAlign = "CENTER"
            story.append(img)
            story.append(Spacer(1, 0.5 * cm))
        except Exception as e:
            logger.warning(f"Logo 加载失败：{e}")

    story.append(Paragraph(school_name, STYLES["subtitle"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("成 绩 分 析 报 告", STYLES["title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY_COLOR, hAlign="CENTER"))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(f"班级：{class_name}", STYLES["subtitle"]))
    story.append(Paragraph(f"生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M')}", STYLES["subtitle"]))
    story.append(Paragraph(f"学生人数：{len(summary_df)} 人    科目数：{len(score_cols)} 科", STYLES["subtitle"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════
    # 第一节：班级整体统计
    # ════════════════════════════════════════════════════
    if selected_outputs.get("class_stats"):
        story.extend(_section_header("一、班级整体统计"))
        story.append(_stats_table(stats_df))
        story.append(Spacer(1, 0.4 * cm))

        # 平均分柱状图
        if "avg_bar" in chart_images:
            try:
                img_buf = io.BytesIO(chart_images["avg_bar"])
                img = Image(img_buf, width=14 * cm, height=7 * cm)
                img.hAlign = "CENTER"
                story.append(img)
                story.append(Paragraph("图1：各科成绩对比（柱=平均分，◆=最高分，✕=最低分）", STYLES["caption"]))
            except Exception as e:
                logger.warning(f"平均分图表加载失败：{e}")

    # ════════════════════════════════════════════════════
    # 第二节：分科成绩分布
    # ════════════════════════════════════════════════════
    if selected_outputs.get("dist_chart"):
        story.extend(_section_header("二、各科成绩分段分布"))

        fig_counter = 2
        for subj in score_cols:
            key = f"dist_{subj}"
            if key in chart_images:
                try:
                    img_buf = io.BytesIO(chart_images[key])
                    img = Image(img_buf, width=14 * cm, height=5.5 * cm)
                    img.hAlign = "CENTER"
                    story.append(img)
                    story.append(Paragraph(f"图{fig_counter}：「{subj}」成绩分段分布", STYLES["caption"]))
                    story.append(Spacer(1, 0.3 * cm))
                    fig_counter += 1
                except Exception as e:
                    logger.warning(f"分布图加载失败：{subj} - {e}")

    # ════════════════════════════════════════════════════
    # 第三节：学生成绩汇总
    # ════════════════════════════════════════════════════
    if selected_outputs.get("student_table"):
        story.append(PageBreak())
        story.extend(_section_header("三、学生成绩汇总（按总分降序）"))
        story.append(_student_table(summary_df, score_cols))

    # ════════════════════════════════════════════════════
    # 第四节：学生个人雷达图（最多展示前6名）
    # ════════════════════════════════════════════════════
    if selected_outputs.get("radar_chart"):
        radar_keys = [k for k in chart_images if k.startswith("radar_")]
        if radar_keys:
            story.append(PageBreak())
            story.extend(_section_header("四、学生个人能力雷达图"))
            story.append(Paragraph("以下雷达图展示每位学生各科成绩均衡性，面积越大代表综合成绩越高。", STYLES["note"]))
            story.append(Spacer(1, 0.3 * cm))

            # 每行排列2个雷达图
            row_imgs = []
            for key in radar_keys[:6]:  # 限制最多6个避免报告过长
                try:
                    img_buf = io.BytesIO(chart_images[key])
                    img = Image(img_buf, width=8 * cm, height=6.5 * cm)
                    row_imgs.append(img)
                    if len(row_imgs) == 2:
                        t = Table([row_imgs], colWidths=[8 * cm, 8 * cm])
                        t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
                        story.append(t)
                        story.append(Spacer(1, 0.2 * cm))
                        row_imgs = []
                except Exception as e:
                    logger.warning(f"雷达图加载失败：{key} - {e}")

            # 处理剩余奇数个
            if row_imgs:
                story.append(row_imgs[0])

    # ════════════════════════════════════════════════════
    # 第五节：教师评价
    # ════════════════════════════════════════════════════
    if teacher_comment and teacher_comment.strip():
        story.append(PageBreak())
        story.extend(_section_header("五、教师评价"))

        if teacher_comment.strip():
            for para in teacher_comment.strip().split("\n"):
                if para.strip():
                    story.append(Paragraph(para.strip(), STYLES["teacher"]))

        story.append(Spacer(1, 1.5 * cm))
        story.append(HRFlowable(width="40%", thickness=0.8, color=BORDER_COLOR, hAlign="RIGHT"))
        story.append(Paragraph(f"教师签名：____________　　日期：____年____月____日", STYLES["body"]))

    # ── 构建 PDF ──
    try:
        doc.build(story)
        logger.info("PDF 报告生成成功")
    except Exception as e:
        logger.error(f"PDF 构建失败：{e}")
        raise RuntimeError(f"PDF 生成失败：{e}") from e

    buf.seek(0)
    return buf.read()
