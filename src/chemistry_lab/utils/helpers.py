#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
辅助工具函数

提供系统中常用的辅助函数。
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

from .logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
        
    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        解析后的字典
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式错误
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON文件不存在: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"成功加载JSON文件: {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON文件格式错误: {file_path}, 错误: {e}")
        raise


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: JSON文件路径
        indent: 缩进空格数
    """
    file_path = Path(file_path)
    
    # 确保目录存在
    ensure_dir(file_path.parent)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        logger.debug(f"成功保存JSON文件: {file_path}")
    except Exception as e:
        logger.error(f"保存JSON文件失败: {file_path}, 错误: {e}")
        raise


def generate_id(prefix: str = "") -> str:
    """
    生成唯一ID
    
    Args:
        prefix: ID前缀
        
    Returns:
        唯一ID字符串
    """
    unique_id = str(uuid.uuid4()).replace("-", "")
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def format_timestamp(timestamp: Optional[float] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    格式化时间戳
    
    Args:
        timestamp: 时间戳，如果为None则使用当前时间
        format_str: 时间格式字符串
        
    Returns:
        格式化后的时间字符串
    """
    if timestamp is None:
        dt = datetime.now()
    else:
        dt = datetime.fromtimestamp(timestamp)
    
    return dt.strftime(format_str)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 除零时的默认值
        
    Returns:
        除法结果或默认值
    """
    if denominator == 0:
        logger.warning(f"除零操作: {numerator} / {denominator}, 返回默认值: {default}")
        return default
    return numerator / denominator


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    将值限制在指定范围内
    
    Args:
        value: 输入值
        min_value: 最小值
        max_value: 最大值
        
    Returns:
        限制后的值
    """
    return max(min_value, min(value, max_value))


def calculate_percentage(part: float, total: float) -> float:
    """
    计算百分比
    
    Args:
        part: 部分值
        total: 总值
        
    Returns:
        百分比 (0-100)
    """
    if total == 0:
        return 0.0
    return (part / total) * 100.0


def format_duration(seconds: float) -> str:
    """
    格式化持续时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串 (如: "1小时23分45秒")
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}分{remaining_seconds:.1f}秒"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}小时{remaining_minutes}分{remaining_seconds:.1f}秒"