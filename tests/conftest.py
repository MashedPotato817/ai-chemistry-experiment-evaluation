#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest配置文件

提供测试夹具和公共配置。
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

# 添加项目根目录到Python路径
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.chemistry_lab.config import Config


@pytest.fixture(scope="session")
def temp_dir():
    """创建临时目录用于测试"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def test_config(temp_dir):
    """创建测试配置"""
    config = Config(
        project_root=temp_dir,
        data_dir=temp_dir / "data",
        models_dir=temp_dir / "models", 
        logs_dir=temp_dir / "logs",
        debug=True,
        log_level="DEBUG",
    )
    
    # 创建测试目录
    config.create_directories()
    
    return config


@pytest.fixture(scope="function")
def mock_database():
    """模拟数据库连接"""
    mock_db = Mock()
    mock_db.execute.return_value = Mock()
    mock_db.commit.return_value = None
    mock_db.rollback.return_value = None
    mock_db.close.return_value = None
    return mock_db


@pytest.fixture(scope="function")
def sample_user_data():
    """示例用户数据"""
    return {
        "username": "test_user",
        "password": "test_password_123",
        "email": "test@example.com",
    }


@pytest.fixture(scope="function")
def sample_experiment_data():
    """示例实验数据"""
    return {
        "experiment_type": "双氧水制氧气",
        "start_time": 1708156800.0,  # 2024-02-17 08:00:00
        "end_time": 1708158600.0,    # 2024-02-17 08:30:00
        "total_duration": 1800.0,    # 30分钟
        "steps": [
            {
                "step_name": "准备器材",
                "start_timestamp": 0.0,
                "end_timestamp": 120.0,
                "duration": 120.0,
            },
            {
                "step_name": "取样双氧水",
                "start_timestamp": 120.0,
                "end_timestamp": 300.0,
                "duration": 180.0,
            },
            {
                "step_name": "添加催化剂",
                "start_timestamp": 300.0,
                "end_timestamp": 450.0,
                "duration": 150.0,
            },
        ]
    }


@pytest.fixture(scope="function")
def sample_detection_data():
    """示例检测数据"""
    return [
        {
            "class_id": 0,
            "class_name": "烧杯",
            "confidence": 0.85,
            "bbox": [100, 100, 200, 200],
            "timestamp": 1708156800.0,
        },
        {
            "class_id": 2,
            "class_name": "滴管",
            "confidence": 0.92,
            "bbox": [300, 150, 350, 250],
            "timestamp": 1708156801.0,
        },
        {
            "class_id": 5,
            "class_name": "手部",
            "confidence": 0.78,
            "bbox": [250, 200, 320, 300],
            "timestamp": 1708156802.0,
        },
    ]