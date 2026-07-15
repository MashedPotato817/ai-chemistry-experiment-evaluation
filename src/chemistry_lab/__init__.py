#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型驱动的智能化学实验熟练度评估实时交互系统

这是一个集成了计算机视觉、自然语言处理和统计分析的智能教育平台，
为化学实验学习者提供个性化的熟练度评估和指导。

主要模块:
- user_management: 用户管理模块
- detection: 时序行为检测模块 
- interaction: 实时交互评估模块
- scoring: 评分引擎
- reporting: 智能报告与推荐模块
- ui: Web用户界面
- database: 数据库访问层
- config: 配置管理
- utils: 工具函数
"""

__version__ = "0.1.0"
__author__ = "Chemistry Lab Assessment Team"
__email__ = "team@chemistry-lab.ai"

# 导出主要类和函数
from .config import Config
from .database import DatabaseManager
from .user_management import UserManager, User
from .detection import YOLODetector, VideoProcessor, Detection, Action, ExperimentStep
from .llm import LLMInteractionService
from .scoring import ScoringEngine

__all__ = [
    "Config",
    "DatabaseManager", 
    "UserManager",
    "User",
    "TemporalActionDetector",
    "Detection",
    "Action", 
    "ExperimentStep",
    "LLMInteractionService",
    "ExperimentContext",
    "PerformanceAnalysis",
    "ScoringEngine",
    "ScoreResult",
    "ReportGenerator",
    "RecommendationEngine",
]