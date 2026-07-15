#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志管理模块

提供统一的日志配置和管理功能。
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger

from ..config import config


def setup_logger(
    log_level: str = None,
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    format_string: str = None,
) -> None:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        rotation: 日志轮转大小
        retention: 日志保留时间
        format_string: 日志格式字符串
    """
    # 移除默认处理器
    logger.remove()
    
    # 使用配置中的默认值
    log_level = log_level or config.log_level
    format_string = format_string or config.log_format
    log_file = log_file or (config.logs_dir / "chemistry_lab.log")
    
    # 控制台输出
    logger.add(
        sys.stdout,
        level=log_level,
        format=format_string,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # 文件输出
    if log_file:
        # 确保日志目录存在
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_file),
            level=log_level,
            format=format_string,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            backtrace=True,
            diagnose=True,
        )
    
    logger.info(f"日志系统初始化完成 - 级别: {log_level}, 文件: {log_file}")


def get_logger(name: str = None):
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        配置好的日志记录器
    """
    if name:
        return logger.bind(name=name)
    return logger


# 默认初始化日志系统
setup_logger()