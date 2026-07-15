#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统配置管理

提供配置加载、验证和管理功能，支持环境变量和配置文件。
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """数据库配置"""
    
    # SQLite数据库文件路径
    database_url: str = Field(
        default="sqlite:///chemistry_lab.db",
        description="数据库连接URL"
    )
    
    # 连接池配置
    pool_size: int = Field(default=10, description="连接池大小")
    max_overflow: int = Field(default=20, description="最大溢出连接数")
    pool_timeout: int = Field(default=30, description="连接池超时时间(秒)")
    
    # 数据库选项
    echo: bool = Field(default=False, description="是否打印SQL语句")
    echo_pool: bool = Field(default=False, description="是否打印连接池信息")
    
    class Config:
        env_prefix = "DB_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class LLMConfig(BaseSettings):
    """大模型API配置"""
    
    # API配置
    api_key: Optional[str] = Field(default=None, description="大模型API密钥")
    api_base: str = Field(
        default="https://api.openai.com/v1",
        description="API基础URL"
    )
    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="使用的模型名称"
    )
    
    # 请求配置
    max_tokens: int = Field(default=1000, description="最大token数")
    temperature: float = Field(default=0.7, description="生成温度")
    timeout: int = Field(default=30, description="请求超时时间(秒)")
    max_retries: int = Field(default=3, description="最大重试次数")
    
    # 速率限制
    requests_per_minute: int = Field(default=60, description="每分钟最大请求数")
    
    class Config:
        env_prefix = "LLM_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class YOLOConfig(BaseSettings):
    """YOLO模型配置"""
    
    # 模型文件路径
    model_path: str = Field(
        default="models/yolov8n_chemistry.pt",
        description="YOLO模型文件路径"
    )
    
    # 检测配置
    confidence_threshold: float = Field(
        default=0.5,
        description="置信度阈值"
    )
    iou_threshold: float = Field(
        default=0.45,
        description="IoU阈值"
    )
    max_detections: int = Field(
        default=100,
        description="最大检测数量"
    )
    
    # 性能配置
    device: str = Field(default="cpu", description="推理设备 (cpu/cuda)")
    batch_size: int = Field(default=1, description="批处理大小")
    input_size: int = Field(default=640, description="输入图像尺寸")
    
    # 检测类别（与 fs_s_best.pt 模型对应）
    class_names: Dict[int, str] = Field(
        default={
            0:  "锥形瓶",
            1:  "双孔橡胶塞",
            2:  "直角导管",
            3:  "橡胶管",
            4:  "集气瓶",
            5:  "玻璃片",
            6:  "止水夹",
            7:  "长颈漏斗",
            8:  "废液缸",
            9:  "酒精灯",
            10: "火柴",
            11: "洗瓶",
            12: "熄灭木条",
            13: "燃烧木条",
            14: "二氧化锰固体",
            15: "药勺",
            16: "手",
            17: "水柱",
            18: "气泡",
        },
        description="检测类别映射"
    )
    
    class Config:
        env_prefix = "YOLO_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class UIConfig(BaseSettings):
    """用户界面配置"""
    
    # Streamlit配置
    page_title: str = Field(
        default="智能化学实验评估系统",
        description="页面标题"
    )
    page_icon: str = Field(default="🧪", description="页面图标")
    layout: str = Field(default="wide", description="页面布局")
    
    # 服务器配置
    host: str = Field(default="localhost", description="服务器主机")
    port: int = Field(default=8501, description="服务器端口")
    
    # 视频流配置
    camera_index: int = Field(default=0, description="摄像头索引")
    video_width: int = Field(default=640, description="视频宽度")
    video_height: int = Field(default=480, description="视频高度")
    fps: int = Field(default=30, description="帧率")
    
    # 界面主题
    theme: str = Field(default="light", description="界面主题")
    primary_color: str = Field(default="#1f77b4", description="主色调")
    
    class Config:
        env_prefix = "UI_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class Config(BaseSettings):
    """主配置类"""
    
    # 应用基本信息
    app_name: str = Field(
        default="智能化学实验评估系统",
        description="应用名称"
    )
    app_version: str = Field(default="0.1.0", description="应用版本")
    debug: bool = Field(default=False, description="调试模式")
    
    # 项目路径
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent,
        description="项目根目录"
    )
    
    # 数据目录
    data_dir: Path = Field(
        default_factory=lambda: Path("data"),
        description="数据目录"
    )
    models_dir: Path = Field(
        default_factory=lambda: Path("models"),
        description="模型目录"
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path("logs"),
        description="日志目录"
    )
    
    # 子配置
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    yolo: YOLOConfig = Field(default_factory=YOLOConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    
    # 安全配置
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="JWT密钥"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT算法")
    jwt_expiration_hours: int = Field(default=24, description="JWT过期时间(小时)")
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="日志格式"
    )
    
    @field_validator("project_root", mode="before")
    @classmethod
    def resolve_project_root(cls, v):
        """解析项目根目录"""
        if isinstance(v, str):
            return Path(v).resolve()
        return v.resolve() if isinstance(v, Path) else v
    
    @field_validator("data_dir", "models_dir", "logs_dir", mode="before")
    @classmethod
    def resolve_relative_paths(cls, v, info):
        """解析相对路径"""
        if isinstance(v, str):
            v = Path(v)
        
        if not v.is_absolute():
            project_root = info.data.get("project_root", Path.cwd())
            return project_root / v
        return v
    
    def create_directories(self):
        """创建必要的目录"""
        directories = [
            self.data_dir,
            self.models_dir, 
            self.logs_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# 全局配置实例
config = Config()

# 创建必要的目录
config.create_directories()