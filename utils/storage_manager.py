"""
storage_manager.py - 本地存储管理模块

负责管理临时文件的创建、读写和清理，
文件命名格式为：{年}{月}{日}_{小时}{分钟}_{随机4位序列码}
"""

import os
import json
import random
import string
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ─── 常量定义 ───────────────────────────────────────────────
TEMP_DIR = Path("temp_cache")           # 临时文件根目录
FILE_EXT = ".cache"                     # 缓存文件扩展名
LOG_FILE = "app.log"                    # 日志文件名
RANDOM_CODE_LENGTH = 4                  # 随机序列码长度
DATE_FORMAT = "%Y%m%d_%H%M"            # 文件名日期格式

# ─── 日志配置 ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _ensure_temp_dir() -> Path:
    """
    确保临时目录存在，不存在则创建。

    Returns:
        Path: 临时目录的 Path 对象
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DIR


def generate_cache_filename() -> str:
    """
    生成符合规范的缓存文件名。

    格式：{年}{月}{日}_{小时}{分钟}_{随机4位序列码}
    示例：20240326_1430_A3K9

    Returns:
        str: 不含扩展名的文件名字符串
    """
    timestamp = datetime.now().strftime(DATE_FORMAT)
    # 随机生成由大写字母和数字组成的4位序列码
    random_code = "".join(
        random.choices(string.ascii_uppercase + string.digits, k=RANDOM_CODE_LENGTH)
    )
    return f"{timestamp}_{random_code}"


def save_session_data(data: Any, session_key: str = "current") -> str:
    """
    将会话数据序列化并保存到临时缓存文件。

    Args:
        data (Any): 需要持久化的数据对象（DataFrame、字典等）
        session_key (str): 会话标识键，默认为 "current"

    Returns:
        str: 保存的文件完整路径字符串

    Raises:
        IOError: 文件写入失败时抛出
    """
    _ensure_temp_dir()
    filename = generate_cache_filename()
    filepath = TEMP_DIR / f"{filename}_{session_key}{FILE_EXT}"

    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"会话数据已保存：{filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"保存会话数据失败：{e}")
        raise IOError(f"无法保存缓存文件：{e}") from e


def load_session_data(filepath: str) -> Any:
    """
    从指定缓存文件加载会话数据。

    Args:
        filepath (str): 缓存文件的完整路径

    Returns:
        Any: 反序列化后的数据对象

    Raises:
        FileNotFoundError: 文件不存在时抛出
        IOError: 文件读取失败时抛出
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"缓存文件不存在：{filepath}")
        raise FileNotFoundError(f"缓存文件不存在：{filepath}")

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"会话数据已加载：{filepath}")
        return data
    except Exception as e:
        logger.error(f"加载会话数据失败：{e}")
        raise IOError(f"无法读取缓存文件：{e}") from e


def save_metadata(meta: dict, filepath: str) -> None:
    """
    将元信息以 JSON 格式附加保存（便于调试和追踪）。

    Args:
        meta (dict): 元信息字典，如上传时间、文件名等
        filepath (str): 对应缓存文件路径（用于生成同名 .meta.json）
    """
    meta_path = Path(filepath).with_suffix(".meta.json")
    meta["saved_at"] = datetime.now().isoformat()
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"元信息已保存：{meta_path}")
    except Exception as e:
        logger.warning(f"元信息保存失败（非致命）：{e}")


def list_cache_files() -> list[Path]:
    """
    列出临时目录中所有缓存文件，按修改时间倒序排列。

    Returns:
        list[Path]: Path 对象列表
    """
    _ensure_temp_dir()
    files = sorted(
        TEMP_DIR.glob(f"*{FILE_EXT}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files


def cleanup_old_caches(keep_latest: int = 10) -> int:
    """
    清理旧缓存文件，仅保留最新的若干个。

    Args:
        keep_latest (int): 保留最新文件的数量，默认保留10个

    Returns:
        int: 被删除的文件数量
    """
    files = list_cache_files()
    to_delete = files[keep_latest:]
    deleted = 0
    for f in to_delete:
        try:
            f.unlink()
            # 同时删除对应的 meta 文件
            meta_f = f.with_suffix(".meta.json")
            if meta_f.exists():
                meta_f.unlink()
            deleted += 1
        except Exception as e:
            logger.warning(f"删除旧缓存失败：{f} - {e}")
    if deleted:
        logger.info(f"已清理 {deleted} 个旧缓存文件")
    return deleted
