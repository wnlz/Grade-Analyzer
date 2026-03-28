import io
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── 将 utils 加入路径 ──
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_processor import (
    read_excel,
    validate_and_clean,
    compute_class_stats,
    compute_all_distributions,
    compute_student_summary,
    get_score_columns,
    set_subject_score_range,
    get_all_subject_ranges,
    delete_subject_score_range,
    SUBJECT_SCORE_RANGES,
)
from utils.visualizer import (
    plot_score_distribution,
    plot_all_distributions,
    plot_radar_chart,
    plot_class_avg_bar,
    plot_score_heatmap,
    mpl_distribution_bar,
    mpl_radar_chart,
    mpl_avg_bar,
)
from utils.pdf_generator import generate_report
from utils.storage_manager import (
    save_session_data,
    load_session_data,
    save_metadata,
    cleanup_old_caches,
    logger as storage_logger,
)

# ─── 常量定义 ───────────────────────────────────────────────
APP_TITLE    = "教师成绩分析工具"
APP_ICON     = "📊"
VERSION      = "1.0.1"
PRIMARY_COLOR = "#2E86AB"
CSS_PATH     = Path(__file__).parent / "assets" / "styles.css"
LOGO_PATH    = Path(__file__).parent / "assets" / "logo.png"

# ─── 日志配置 ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 页面配置（必须是第一个 Streamlit 调用）
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# 暂时忽略
st.markdown(
    """
    <style>
    .stAppHeader {
        display: none;
    }
    div[data-testid="stSidebarHeader"] {
        display: none;
    }
    div[data-testid="stMainBlockContainer"] {
        padding-top: 12px;
        padding-bottom: 96px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# ══════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════

def load_css():
    """加载自定义 CSS 样式文件并注入到页面。"""
    if CSS_PATH.exists():
        with open(CSS_PATH, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header():
    """渲染顶部标题横幅。"""
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{APP_ICON} {APP_TITLE}</h1>
            <p>帮助教师快速完成考后数据分析 · 支持 Excel 上传 · 智能统计与可视化 · 一键生成 PDF 报告</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, suffix: str = "") -> str:
    """
    生成单个统计指标卡的 HTML 片段。

    Args:
        label (str): 指标名称
        value (str): 主要数值
        suffix (str): 单位后缀

    Returns:
        str: HTML 字符串
    """
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}<span style="font-size:1rem;font-weight:400">{suffix}</span></div>
        <div class="metric-label">{label}</div>
    </div>
    """


def show_validation_messages(errors: list[str], warnings: list[str]):
    """
    在页面上展示数据验证的错误和警告信息。

    Args:
        errors (list[str]): 阻断性错误列表
        warnings (list[str]): 非阻断性警告列表
    """
    for msg in errors:
        st.error(f"❌ {msg}")
    for msg in warnings:
        st.warning(f"⚠️ {msg}")
    if errors:
        st.stop()


def get_chart_images(
    stats_df: pd.DataFrame,
    df: pd.DataFrame,
    score_cols: list[str],
    all_dist: dict,
    selected_students: Optional[list[str]] = None,
) -> dict[str, bytes]:
    """
    批量生成用于 PDF 嵌入的 Matplotlib 静态图表字节流。

    Args:
        stats_df: 统计表
        df: 学生汇总表
        score_cols: 成绩列名
        all_dist: 各科分布数据
        selected_students: 需要生成雷达图的学号列表，None 表示全部（最多6个）

    Returns:
        dict[str, bytes]: 图表名称 -> PNG 字节流
    """
    images: dict[str, bytes] = {}

    # 平均分柱图
    with st.spinner("正在生成平均分图表…"):
        try:
            images["avg_bar"] = mpl_avg_bar(stats_df)
        except Exception as e:
            logger.warning(f"平均分图生成失败：{e}")

    # 各科分布图
    for subj, dist in all_dist.items():
        try:
            images[f"dist_{subj}"] = mpl_distribution_bar(dist, subj)
        except Exception as e:
            logger.warning(f"分布图生成失败：{subj} - {e}")

    # 雷达图
    candidates = df.head(6) if selected_students is None else df[df["学号"].isin(selected_students)]
    for _, row in candidates.iterrows():
        try:
            images[f"radar_{row['学号']}"] = mpl_radar_chart(row, score_cols)
        except Exception as e:
            logger.warning(f"雷达图生成失败：{row.get('学号')} - {e}")

    return images


# ══════════════════════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """
    渲染侧边栏，收集报告配置参数。

    Returns:
        dict: 包含 school_name、class_name、teacher_comment、
              selected_outputs、logo_path 的配置字典
    """
    with st.sidebar:
        st.markdown("## ⚙️ 报告配置")
        st.divider()
        st.markdown("### 📚 科目分数范围设置")
        st.caption("可自定义各科目的分数范围，语数英默认为0-150分，其他科目默认为0-100分")
        
        # 显示当前配置
        current_ranges = get_all_subject_ranges()
        
        # 编辑科目范围
        edit_mode = st.checkbox("✏️ 编辑科目分数范围", key="edit_ranges")
        
        if edit_mode:
            st.markdown("**添加/修改科目范围**")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                new_subject = st.text_input("科目名称", key="new_subject", placeholder="如：物理")
            with col2:
                new_min = st.number_input("最低分", value=0, key="new_min", min_value=0, max_value=200)
            with col3:
                new_max = st.number_input("最高分", value=100, key="new_max", min_value=1, max_value=200)
            
            if st.button("➕ 添加/更新科目范围", use_container_width=True):
                if new_subject.strip():
                    set_subject_score_range(new_subject.strip(), int(new_min), int(new_max))
                    st.success(f"已设置「{new_subject}」分数范围为 [{new_min}, {new_max}]")
                    st.rerun()
                else:
                    st.warning("请输入科目名称")
            
            st.divider()
            st.markdown("**当前科目范围配置**")
            
            # 显示所有配置
            ranges_df = pd.DataFrame([
                {"科目": subj, "最低分": min_val, "最高分": max_val}
                for subj, (min_val, max_val) in current_ranges.items()
            ])
            st.dataframe(ranges_df, use_container_width=True, hide_index=True)
            
            # 删除科目
            if current_ranges:
                st.markdown("**删除科目配置**")
                subject_to_delete = st.selectbox(
                    "选择要删除的科目",
                    options=list(current_ranges.keys()),
                    key="delete_subject"
                )
                if st.button("🗑️ 删除科目范围", use_container_width=True):
                    delete_subject_score_range(subject_to_delete)
                    st.success(f"已删除「{subject_to_delete}」的分数范围配置")
                    st.rerun()
        else:
            # 显示当前配置摘要
            with st.expander("查看当前科目分数范围配置"):
                ranges_text = ""
                for subj, (min_val, max_val) in current_ranges.items():
                    ranges_text += f"• **{subj}**: {min_val}-{max_val}分\n\n"
                st.markdown(ranges_text)
                st.caption("勾选「编辑科目分数范围」可修改配置")

        st.divider()
        st.markdown("### 📋 选择输出内容")
        output_options = {
            "class_stats":   "🔎 各科成绩总览",
            "dist_chart":    "📊 分数段分布图",
            "radar_chart":   "🕸️ 学生雷达图",
            "heatmap":       "🌡️ 成绩热力图",
            "student_table": "📝 学生成绩汇总表",
        }
        selected_outputs = {}
        for key, label in output_options.items():
            selected_outputs[key] = st.checkbox(label, value=True, key=f"out_{key}")

        st.divider()
        st.markdown("### 🏫 学校-班级配置（可选）")
        school_name = st.text_input(
            "🏘️ 学校名称",
            value=st.session_state.get("school_name", ""),
            placeholder="请输入学校名称",
        )
        class_name = st.text_input(
            "🎓 班级",
            value=st.session_state.get("class_name", ""),
            placeholder="如：高三(2)班",
        )
        st.markdown("🖼️ Logo")
        if Path(LOGO_PATH).exists():
                st.markdown("**（已存在默认Logo）**")
                st.image(str(LOGO_PATH), width=100)
        logo_file = st.file_uploader(
            "上传 PNG/JPG 格式 Logo",
            type=["png", "jpg", "jpeg"],
            key="logo_upload",
            label_visibility="collapsed",
            max_upload_size=20
        )
        logo_path = None
        if logo_file:
            logo_dir = Path("assets")
            logo_dir.mkdir(exist_ok=True)
            logo_path = str(logo_dir / "logo_upload.png")
            with open(logo_path, "wb") as f:
                f.write(logo_file.read())
            st.success("Logo 已上传 ✓")

        st.divider()
        st.markdown("### 💬 教师评价（可选）")
        teacher_comment = st.text_area(
            "在此输入教师评价内容（将嵌入 PDF 报告）",
            value=st.session_state.get("teacher_comment", ""),
            placeholder="请对本次考试情况进行总结性评价…",
            height=140,
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(f"📦 版本 {VERSION}")

        # 保存到 session_state
        st.session_state["school_name"]    = school_name
        st.session_state["class_name"]     = class_name
        st.session_state["teacher_comment"]= teacher_comment

    return {
        "school_name":      school_name,
        "class_name":       class_name,
        "teacher_comment":  teacher_comment,
        "selected_outputs": selected_outputs,
        "logo_path":        logo_path or (str(LOGO_PATH) if Path(LOGO_PATH).exists() else None),
    }


# ══════════════════════════════════════════════════════════════
# 各功能 Tab 渲染
# ══════════════════════════════════════════════════════════════

def tab_class_stats(stats_df: pd.DataFrame, score_cols: list[str]):
    '''
    """
    渲染「各科成绩总览」标签页。

    Args:
        stats_df: compute_class_stats 返回的统计表
        score_cols: 成绩列名列表
    """
    st.markdown("### 📊 各科成绩统计概览")

    # 每科一组指标卡（每行最多4列）
    for subj in score_cols:
        st.markdown(f"#### 📚 {subj}")
        cols = st.columns(5)
        indicators = [
            ("平均分", f"{stats_df.loc['平均分', subj]:.1f}" if not np.isnan(stats_df.loc['平均分', subj]) else "-", ""),
            ("最高分", str(int(stats_df.loc['最高分', subj])) if not np.isnan(stats_df.loc['最高分', subj]) else "-", ""),
            ("最低分", str(int(stats_df.loc['最低分', subj])) if not np.isnan(stats_df.loc['最低分', subj]) else "-", ""),
            ("标准差",  f"{stats_df.loc['标准差', subj]:.2f}"  if not np.isnan(stats_df.loc['标准差', subj])  else "-", ""),
            ("有效人数", str(int(stats_df.loc['有效人数', subj])), " 人"),
        ]
        for col, (label, val, suffix) in zip(cols, indicators):
            col.markdown(render_metric_card(label, val, suffix), unsafe_allow_html=True)
        st.markdown("")  # 间距
    '''

    # 汇总对比图
    st.plotly_chart(plot_class_avg_bar(stats_df), use_container_width=True)

    # 原始统计表（可展开）
    with st.expander("查看完整统计数据表"):
        st.dataframe(stats_df.style.format("{:.2f}", na_rep="-"), use_container_width=True)


def tab_distribution(df: pd.DataFrame, all_dist: dict[str, pd.Series]):
    """
    渲染「分数分布」标签页。

    Args:
        df: 清洗后的 DataFrame
        all_dist: 各科分布数据
    """
    score_cols = list(all_dist.keys())

    st.markdown("### 📈 分数段分布分析")
    st.caption("注：不同科目可能使用不同的分数段区间（如100分制 vs 150分制）")

    # 总览多图
    if len(score_cols) > 1:
        st.plotly_chart(plot_all_distributions(all_dist), use_container_width=True)

    # 单科详细
    st.divider()
    selected_subj = st.selectbox(
        "选择科目查看详细分布",
        options=score_cols,
        key="dist_subject_select",
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(
            plot_score_distribution(all_dist[selected_subj], selected_subj),
            use_container_width=True,
        )
    with col2:
        st.markdown("**分布数据**")
        dist_table = all_dist[selected_subj].reset_index()
        dist_table.columns = ["分数段", "人数"]
        total = dist_table["人数"].sum()
        dist_table["占比"] = dist_table["人数"].apply(
            lambda x: f"{x/total*100:.1f}%" if total > 0 else "0%"
        )
        st.dataframe(dist_table, hide_index=True, use_container_width=True)


def tab_radar(df: pd.DataFrame, score_cols: list[str]):
    """
    渲染「学生雷达图」标签页。

    Args:
        df: 学生汇总 DataFrame（含排名）
        score_cols: 成绩列名列表
    """
    st.markdown("### 🕸️ 学生个人成绩雷达图")
    st.caption("雷达图展示学生各科成绩的均衡性，面积越大、越圆润，说明综合成绩越好。")

    if len(score_cols) < 3:
        st.info("科目数需达到 3 门及以上才能生成有意义的雷达图。")
        return

    # 学生选择器
    student_options = (df["学号"] + " - " + df["姓名"]).tolist()
    selected = st.multiselect(
        "选择学生（可多选，最多同时显示6人）",
        options=student_options,
        default=student_options[:min(4, len(student_options))],
        key="radar_students",
    )

    if not selected:
        st.info("请至少选择一位学生。")
        return

    display_ids = [s.split(" - ")[0] for s in selected[:6]]
    display_df  = df[df["学号"].isin(display_ids)]

    # 每行2个雷达图
    rows = [display_df.iloc[i : i + 2] for i in range(0, len(display_df), 2)]
    for row_df in rows:
        cols = st.columns(2)
        for col, (_, student) in zip(cols, row_df.iterrows()):
            with col:
                st.plotly_chart(
                    plot_radar_chart(student, score_cols),
                    use_container_width=True,
                )


def tab_heatmap(df: pd.DataFrame, score_cols: list[str]):
    """
    渲染「成绩热力图」标签页。

    Args:
        df: 学生汇总 DataFrame
        score_cols: 成绩列名列表
    """
    st.markdown("### 🌡️ 全班成绩热力图")
    st.caption("颜色越深绿代表分数越高，红色代表分数较低，可快速定位薄弱科目和学生。")

    if len(df) > 50:
        st.info(f"【班级共 {len(df)} 人，热力图仅展示前 50 名学生（按总分降序）。】")

    st.plotly_chart(plot_score_heatmap(df, score_cols), use_container_width=True)


def tab_students(summary_df: pd.DataFrame, score_cols: list[str]):
    """
    渲染「学生成绩表」标签页。

    Args:
        summary_df: compute_student_summary 返回的 DataFrame
        score_cols: 成绩列名列表
    """
    st.markdown("### 📝 学生成绩汇总表")

    # 搜索框
    search = st.text_input("🔍 搜索学号或姓名", placeholder="输入关键词实时过滤…", key="student_search")

    display_df = summary_df.copy()
    if search.strip():
        mask = (
            display_df["学号"].str.contains(search, case=False, na=False) |
            display_df["姓名"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]
        st.caption(f"共找到 {len(display_df)} 条匹配记录")

    # 显示列
    show_cols = ["排名", "学号", "姓名"] + score_cols + ["总分", "平均分"]

    # 成绩着色 - 根据不同科目的满分动态调整
    def color_score(val, col_name=None):
        """根据分数区间返回背景色样式，考虑不同科目的满分"""
        if not isinstance(val, (int, float)) or np.isnan(val):
            return ""
        
        # 获取该科目的满分
        max_score = 100
        if col_name and col_name in SUBJECT_SCORE_RANGES:
            _, max_score = SUBJECT_SCORE_RANGES[col_name]
        
        # 按比例计算阈值（60%、75%、85%）
        threshold_60 = max_score * 0.6
        threshold_75 = max_score * 0.75
        threshold_85 = max_score * 0.85
        
        if val < threshold_60:
            return "background-color: #FDECEA; color: #C0392B"
        if val < threshold_75:
            return "background-color: #FEF9E7; color: #B7950B"
        if val < threshold_85:
            return "background-color: #E8F8F5; color: #1E8449"
        return "background-color: #EBF5FB; color: #1A5276"
    
    # 应用样式
    styled = display_df[show_cols].style
    
    # 为每个科目单独着色
    for col in score_cols + ["总分", "平均分"]:
        styled = styled.applymap(
            lambda x, c=col: color_score(x, c),
            subset=[col]
        )
    
    styled = styled.format(
        {col: "{:.1f}" for col in score_cols + ["总分", "平均分"]},
        na_rep="-"
    )
    
    st.dataframe(styled, use_container_width=True, height=500)

    # 下载原始数据
    csv_buf = io.StringIO()
    display_df[show_cols].to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button(
        label="⬇️ 下载当前表格（CSV）",
        data=csv_buf.getvalue().encode("utf-8-sig"),
        file_name="成绩汇总.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    """程序入口：加载样式，渲染界面，处理上传文件并展示分析结果。"""
    load_css()
    render_header()
    config = render_sidebar()

    # ── 步骤指引 ──
    with st.expander("步骤指引"):
        st.markdown(
            """
            <div class="step-guide"><div class="step-num">1</div> 在侧边栏配置科目分数范围（可选）</div>
            <div class="step-guide"><div class="step-num">2</div> 在下方上传 Excel 成绩文件（.xlsx / .xls）</div>
            <div class="step-guide"><div class="step-num">4</div> 在侧边栏选择输出内容，浏览各项分析</div>
            <div class="step-guide"><div class="step-num">6</div> 点击「生成 PDF 报告」一键导出</div>
            """,
            unsafe_allow_html=True,
        )

    # ── 文件上传 ──
    st.markdown("### 📂 上传成绩文件")
    
    # 获取当前科目范围配置用于显示
    current_ranges = get_all_subject_ranges()
    range_text = "、".join([f"{subj}({min_val}-{max_val}分)" for subj, (min_val, max_val) in current_ranges.items()])
    
    st.markdown(
        f"""
        <div class="card">
        <b>格式要求：</b>
        第一列为<b>学号</b>，第二列为<b>姓名</b>，后续列为各科成绩（数值型）<br>
        <b>分数范围：</b> {range_text}<br>
        文件大小 ≤ 20MB，支持 .xlsx 和 .xls 格式。
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "拖拽文件到此处，或点击选择文件",
        type=["xlsx", "xls"],
        key="grade_file",
        label_visibility="collapsed",
        max_upload_size=20
    )

    if uploaded_file is None:
        st.info("👆 请先上传成绩文件以开始分析。")
        return

    # ── 缓存：避免重复处理同一文件 ──
    file_hash = hash(uploaded_file.name + str(uploaded_file.size))
    cache_key  = f"processed_{file_hash}"

    if cache_key not in st.session_state:
        with st.spinner("📖 正在读取和验证数据…"):
            try:
                raw_df, read_warnings = read_excel(uploaded_file)
                clean_df, errors, warnings = validate_and_clean(raw_df)
            except (ValueError, IOError) as e:
                st.error(f"❌ 文件处理失败：{e}")
                logger.error(f"文件处理异常：{e}")
                return

        show_validation_messages(errors, read_warnings + warnings)

        # 计算各类统计数据
        score_cols  = get_score_columns(clean_df)
        stats_df    = compute_class_stats(clean_df)
        all_dist    = compute_all_distributions(clean_df)
        summary_df  = compute_student_summary(clean_df)

        # 持久化到临时缓存
        try:
            cache_path = save_session_data(
                {
                    "clean_df":  clean_df,
                    "stats_df":  stats_df,
                    "all_dist":  all_dist,
                    "summary_df":summary_df,
                    "score_cols":score_cols,
                },
                session_key="grade_session",
            )
            save_metadata(
                {
                    "filename":  uploaded_file.name,
                    "rows":      len(clean_df),
                    "subjects":  score_cols,
                },
                cache_path,
            )
            cleanup_old_caches(keep_latest=10)  # 保持缓存目录整洁
        except Exception as e:
            logger.warning(f"缓存保存失败（非致命）：{e}")

        # 写入 session_state
        st.session_state[cache_key] = {
            "clean_df":  clean_df,
            "stats_df":  stats_df,
            "all_dist":  all_dist,
            "summary_df":summary_df,
            "score_cols":score_cols,
        }
        
        # 显示科目分数范围信息
        subject_ranges_info = []
        for col in score_cols:
            if col in SUBJECT_SCORE_RANGES:
                min_s, max_s = SUBJECT_SCORE_RANGES[col]
                subject_ranges_info.append(f"{col}({min_s}-{max_s}分)")
            else:
                subject_ranges_info.append(f"{col}(0-100分)")
        
        st.success(f"✅ 文件「{uploaded_file.name}」加载成功！共 {len(clean_df)} 名学生，{len(score_cols)} 个科目。")

    # 从 session_state 取出处理结果
    data = st.session_state[cache_key]
    clean_df  = data["clean_df"]
    stats_df  = data["stats_df"]
    all_dist  = data["all_dist"]
    summary_df= data["summary_df"]
    score_cols= data["score_cols"]

    selected_outputs = config["selected_outputs"]

    # ── 分析结果展示（多 Tab） ──
    st.markdown("---")
    st.markdown("## 📈 分析结果")

    active_tabs  = []
    tab_renderers = []

    if selected_outputs.get("class_stats"):
        active_tabs.append("📊 各科成绩总览")
        tab_renderers.append(lambda: tab_class_stats(stats_df, score_cols))
    if selected_outputs.get("dist_chart"):
        active_tabs.append("📈 分数分布")
        tab_renderers.append(lambda: tab_distribution(clean_df, all_dist))
    if selected_outputs.get("radar_chart"):
        active_tabs.append("🕸️ 学生雷达图")
        tab_renderers.append(lambda: tab_radar(summary_df, score_cols))
    if selected_outputs.get("heatmap"):
        active_tabs.append("🌡️ 热力图")
        tab_renderers.append(lambda: tab_heatmap(summary_df, score_cols))
    if selected_outputs.get("student_table"):
        active_tabs.append("📝 成绩汇总")
        tab_renderers.append(lambda: tab_students(summary_df, score_cols))

    if not active_tabs:
        st.info("请在侧边栏勾选至少一项输出内容。")
        return

    tabs = st.tabs(active_tabs)
    for tab, renderer in zip(tabs, tab_renderers):
        with tab:
            renderer()

    # ── PDF 生成区 ──
    st.markdown("---")
    st.markdown("## 📄 生成报告")

    # 根据侧边栏的勾选动态生成报告包含内容说明
    selected_outputs = config["selected_outputs"]
    included_items = []

    if selected_outputs.get("class_stats"):
        included_items.append("班级统计")
    if selected_outputs.get("dist_chart"):
        included_items.append("分布图")
    if selected_outputs.get("student_table"):
        included_items.append("学生汇总表")
    if selected_outputs.get("radar_chart"):
        included_items.append("雷达图（前6名）")
    if config["teacher_comment"] and config["teacher_comment"].strip():
        included_items.append("教师评价")

    if not included_items:
        included_text = "请在侧边栏勾选至少一项输出内容。"
    else:
        included_text = "、".join(included_items)

    col_gen, col_info = st.columns([1, 3])
    with col_gen:
        gen_btn = st.button("🖨️ 生成 PDF 报告", use_container_width=True, type="primary")

    with col_info:
        st.markdown(
            f"""
            <div style="padding:0.7rem 1rem; background:#EFF8FC; border-radius:8px; font-size:0.9rem">
            报告将包含：<b>{included_text}</b>。<br>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if gen_btn:
        progress = st.progress(0, text="正在准备图表…")
        try:
            # 生成静态图表
            chart_images = get_chart_images(stats_df, summary_df, score_cols, all_dist)
            progress.progress(50, text="正在构建 PDF…")

            pdf_bytes = generate_report(
                stats_df       = stats_df,
                summary_df     = summary_df,
                all_dist       = all_dist,
                score_cols     = score_cols,
                chart_images   = chart_images,
                teacher_comment= config["teacher_comment"],
                school_name    = config["school_name"] or "学校",
                class_name     = config["class_name"]  or "班级",
                logo_path      = config["logo_path"],
                selected_outputs = selected_outputs
            )
            progress.progress(100, text="PDF 生成完毕！")

            filename = (
                f"{config['school_name'] or '成绩'}_{config['class_name'] or '班级'}"
                f"_分析报告.pdf"
            )
            st.download_button(
                label     = "⬇️ 下载 PDF 报告",
                data      = pdf_bytes,
                file_name = filename,
                mime      = "application/pdf",
            )
            st.success("✅ PDF 报告已生成，点击上方按钮下载！")
        except Exception as e:
            progress.empty()
            st.error(f"❌ PDF 生成失败：{e}")
            logger.error(f"PDF 生成异常：{e}", exc_info=True)


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()