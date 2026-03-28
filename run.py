"""
run.py - 一键启动脚本

运行此文件将自动安装依赖并打开浏览器页面。
使用方式：python run.py
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

# ─── 常量 ────────────────────────────────────────────────────
PORT        = 8501
HOST        = "localhost"
APP_FILE    = "main.py"
REQUIREMENTS= "requirements.txt"
BROWSER_DELAY = 2.5   # 等待 Streamlit 启动后再打开浏览器的秒数


def install_dependencies():
    """
    自动安装 requirements.txt 中列出的依赖包。
    安装失败时输出错误但不中断启动流程。
    """
    req_path = Path(__file__).parent / REQUIREMENTS
    if not req_path.exists():
        print(f"[警告] 未找到 {REQUIREMENTS}，跳过依赖安装")
        return

    print("📦 正在检查并安装依赖…（首次运行可能需要几分钟）")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_path), "-q"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("✅ 依赖安装完成")
    else:
        print(f"[警告] 部分依赖安装失败：\n{result.stderr[:500]}")


def open_browser_after_delay():
    """在子线程中延迟打开浏览器，等待 Streamlit 服务就绪。"""
    import threading

    def _open():
        time.sleep(BROWSER_DELAY)
        url = f"http://{HOST}:{PORT}"
        print(f"🌐 正在打开浏览器：{url}")
        webbrowser.open(url)

    t = threading.Thread(target=_open, daemon=True)
    t.start()


def run_streamlit():
    """
    启动 Streamlit 应用服务。
    将服务器地址、端口绑定后通过 subprocess 运行。
    """
    app_path = Path(__file__).parent / APP_FILE
    if not app_path.exists():
        print(f"[错误] 未找到主程序文件：{app_path}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        f"--server.port={PORT}",
        f"--server.address={HOST}",
        "--server.headless=true",     # 禁用自动打开（由本脚本控制）
        "--browser.gatherUsageStats=false",
    ]

    print(f"\n🚀 启动教师成绩分析工具…")
    print(f"   地址：http://{HOST}:{PORT}")
    print(f"   按 Ctrl+C 停止服务\n")

    open_browser_after_delay()

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 服务已停止。")


if __name__ == "__main__":
    # 切换到脚本所在目录，确保相对路径正确
    os.chdir(Path(__file__).parent)
    install_dependencies()
    run_streamlit()
