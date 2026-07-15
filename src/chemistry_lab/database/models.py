#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模型定义

定义系统中使用的所有数据表模型。
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, Text, Boolean, DateTime, 
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from ..utils.exceptions import ValidationError

Base = declarative_base()


class User(Base):
    """用户表模型"""
    
    __tablename__ = "users"
    
    # 主键
    user_id = Column(Integer, primary_key=True, autoincrement=True, comment="用户ID")
    
    # 基本信息
    username = Column(String(50), unique=True, nullable=False, comment="用户名")
    password_hash = Column(String(255), nullable=False, comment="密码哈希")
    email = Column(String(100), unique=True, nullable=True, comment="邮箱地址")
    
    # 时间戳
    created_at = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
    last_login = Column(DateTime, nullable=True, comment="最后登录时间")
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 用户状态
    is_active = Column(Boolean, default=True, nullable=False, comment="是否激活")
    is_admin = Column(Boolean, default=False, nullable=False, comment="是否管理员")
    
    # 关联关系
    experiment_sessions = relationship("ExperimentSession", back_populates="user", cascade="all, delete-orphan")
    
    # 索引
    __table_args__ = (
        Index("idx_username", "username"),
        Index("idx_email", "email"),
        Index("idx_created_at", "created_at"),
    )
    
    @validates("username")
    def validate_username(self, key, username):
        """验证用户名"""
        if not username or len(username.strip()) < 3:
            raise ValidationError("用户名长度至少3个字符", field="username", value=username)
        if len(username) > 50:
            raise ValidationError("用户名长度不能超过50个字符", field="username", value=username)
        return username.strip()
    
    @validates("email")
    def validate_email(self, key, email):
        """验证邮箱地址"""
        if email:
            import re
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(pattern, email):
                raise ValidationError("邮箱地址格式无效", field="email", value=email)
        return email
    
    def __repr__(self):
        return f"<User(id={self.user_id}, username='{self.username}')>"


class ExperimentSession(Base):
    """实验会话表模型"""
    
    __tablename__ = "experiment_sessions"
    
    # 主键
    session_id = Column(Integer, primary_key=True, autoincrement=True, comment="会话ID")
    
    # 外键
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, comment="用户ID")
    
    # 实验信息
    experiment_type = Column(String(100), nullable=False, comment="实验类型")
    experiment_name = Column(String(200), nullable=True, comment="实验名称")
    
    # 时间信息
    start_time = Column(DateTime, nullable=False, comment="开始时间")
    end_time = Column(DateTime, nullable=True, comment="结束时间")
    total_duration = Column(Float, nullable=True, comment="总持续时间(秒)")
    
    # 评分信息
    s1_score = Column(Float, nullable=True, comment="S1时长得分")
    s2_score = Column(Float, nullable=True, comment="S2表现得分")
    final_score = Column(Float, nullable=True, comment="最终得分")
    percentile_rank = Column(Float, nullable=True, comment="百分位排名")
    
    # 文件路径
    video_path = Column(String(500), nullable=True, comment="视频文件路径")
    report_path = Column(String(500), nullable=True, comment="报告文件路径")
    
    # 状态信息
    status = Column(String(20), default="active", nullable=False, comment="会话状态")
    notes = Column(Text, nullable=True, comment="备注信息")
    
    # 时间戳
    created_at = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 关联关系
    user = relationship("User", back_populates="experiment_sessions")
    experiment_steps = relationship("ExperimentStep", back_populates="session", cascade="all, delete-orphan")
    interaction_logs = relationship("InteractionLog", back_populates="session", cascade="all, delete-orphan")
    
    # 索引
    __table_args__ = (
        Index("idx_user_id", "user_id"),
        Index("idx_experiment_type", "experiment_type"),
        Index("idx_start_time", "start_time"),
        Index("idx_status", "status"),
        Index("idx_final_score", "final_score"),
    )
    
    @validates("experiment_type")
    def validate_experiment_type(self, key, experiment_type):
        """验证实验类型"""
        if not experiment_type or len(experiment_type.strip()) == 0:
            raise ValidationError("实验类型不能为空", field="experiment_type", value=experiment_type)
        return experiment_type.strip()
    
    @validates("status")
    def validate_status(self, key, status):
        """验证状态"""
        valid_statuses = ["active", "completed", "cancelled", "error"]
        if status not in valid_statuses:
            raise ValidationError(f"无效的状态值: {status}", field="status", value=status)
        return status
    
    @property
    def is_completed(self) -> bool:
        """检查会话是否已完成"""
        return self.status == "completed" and self.end_time is not None
    
    def calculate_duration(self) -> Optional[float]:
        """计算实验持续时间"""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds()
        return None
    
    def __repr__(self):
        return f"<ExperimentSession(id={self.session_id}, type='{self.experiment_type}', user_id={self.user_id})>"


class ExperimentStep(Base):
    """实验步骤表模型"""
    
    __tablename__ = "experiment_steps"
    
    # 主键
    step_id = Column(Integer, primary_key=True, autoincrement=True, comment="步骤ID")
    
    # 外键
    session_id = Column(Integer, ForeignKey("experiment_sessions.session_id"), nullable=False, comment="会话ID")
    
    # 步骤信息
    step_name = Column(String(100), nullable=False, comment="步骤名称")
    step_description = Column(Text, nullable=True, comment="步骤描述")
    
    # 时间信息
    start_timestamp = Column(Float, nullable=False, comment="开始时间戳(相对于实验开始)")
    end_timestamp = Column(Float, nullable=False, comment="结束时间戳(相对于实验开始)")
    duration = Column(Float, nullable=False, comment="持续时间(秒)")
    pause_duration = Column(Float, default=0.0, nullable=False, comment="停顿时间(秒)")
    
    # 顺序和状态
    sequence_order = Column(Integer, nullable=False, comment="步骤顺序")
    is_correct = Column(Boolean, nullable=True, comment="步骤是否正确")
    
    # 检测数据
    detected_objects = Column(Text, nullable=True, comment="检测到的物体(JSON格式)")
    anomalies = Column(Text, nullable=True, comment="异常操作记录(JSON格式)")
    
    # 时间戳
    created_at = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
    
    # 关联关系
    session = relationship("ExperimentSession", back_populates="experiment_steps")
    
    # 索引
    __table_args__ = (
        Index("idx_step_session_id", "session_id"),
        Index("idx_step_name", "step_name"),
        Index("idx_sequence_order", "sequence_order"),
        Index("idx_start_timestamp", "start_timestamp"),
        UniqueConstraint("session_id", "sequence_order", name="uq_session_sequence"),
    )
    
    @validates("step_name")
    def validate_step_name(self, key, step_name):
        """验证步骤名称"""
        if not step_name or len(step_name.strip()) == 0:
            raise ValidationError("步骤名称不能为空", field="step_name", value=step_name)
        return step_name.strip()
    
    @validates("start_timestamp", "end_timestamp")
    def validate_timestamps(self, key, timestamp):
        """验证时间戳"""
        if timestamp < 0:
            raise ValidationError("时间戳不能为负数", field=key, value=timestamp)
        return timestamp
    
    @validates("sequence_order")
    def validate_sequence_order(self, key, sequence_order):
        """验证步骤顺序"""
        if sequence_order < 1:
            raise ValidationError("步骤顺序必须大于0", field="sequence_order", value=sequence_order)
        return sequence_order
    
    def __repr__(self):
        return f"<ExperimentStep(id={self.step_id}, name='{self.step_name}', session_id={self.session_id})>"


class InteractionLog(Base):
    """交互记录表模型"""
    
    __tablename__ = "interaction_logs"
    
    # 主键
    log_id = Column(Integer, primary_key=True, autoincrement=True, comment="日志ID")
    
    # 外键
    session_id = Column(Integer, ForeignKey("experiment_sessions.session_id"), nullable=False, comment="会话ID")
    
    # 交互信息
    timestamp = Column(Float, nullable=False, comment="时间戳(相对于实验开始)")
    interaction_type = Column(String(50), nullable=False, comment="交互类型")
    
    # 内容信息
    user_input = Column(Text, nullable=True, comment="用户输入")
    system_response = Column(Text, nullable=True, comment="系统响应")
    context_data = Column(Text, nullable=True, comment="上下文数据(JSON格式)")
    
    # 元数据
    response_time = Column(Float, nullable=True, comment="响应时间(毫秒)")
    confidence_score = Column(Float, nullable=True, comment="置信度分数")
    
    # 时间戳
    created_at = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
    
    # 关联关系
    session = relationship("ExperimentSession", back_populates="interaction_logs")
    
    # 索引
    __table_args__ = (
        Index("idx_log_session_id", "session_id"),
        Index("idx_interaction_type", "interaction_type"),
        Index("idx_log_timestamp", "timestamp"),
        Index("idx_log_created_at", "created_at"),
    )
    
    @validates("interaction_type")
    def validate_interaction_type(self, key, interaction_type):
        """验证交互类型"""
        valid_types = [
            "question", "guidance", "warning", "error", 
            "feedback", "recommendation", "voice_input", "text_input"
        ]
        if interaction_type not in valid_types:
            raise ValidationError(f"无效的交互类型: {interaction_type}", field="interaction_type", value=interaction_type)
        return interaction_type
    
    def __repr__(self):
        return f"<InteractionLog(id={self.log_id}, type='{self.interaction_type}', session_id={self.session_id})>"


class ExperimentStatistics(Base):
    """实验统计表模型"""
    
    __tablename__ = "experiment_statistics"
    
    # 主键
    stat_id = Column(Integer, primary_key=True, autoincrement=True, comment="统计ID")
    
    # 实验类型
    experiment_type = Column(String(100), nullable=False, comment="实验类型")
    
    # 统计数据
    mean_duration = Column(Float, nullable=False, comment="平均持续时间(秒)")
    std_deviation = Column(Float, nullable=False, comment="标准差(秒)")
    sample_count = Column(Integer, nullable=False, comment="样本数量")
    
    # 分数统计
    mean_score = Column(Float, nullable=True, comment="平均分数")
    score_std_deviation = Column(Float, nullable=True, comment="分数标准差")
    
    # 步骤统计
    typical_steps = Column(Text, nullable=True, comment="典型步骤序列(JSON格式)")
    step_durations = Column(Text, nullable=True, comment="步骤持续时间统计(JSON格式)")
    
    # 时间戳
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, comment="最后更新时间")
    created_at = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
    
    # 索引
    __table_args__ = (
        Index("idx_standard_experiment_type", "experiment_type"),
        Index("idx_standard_last_updated", "last_updated"),
        UniqueConstraint("experiment_type", name="uq_experiment_type"),
    )
    
    @validates("experiment_type")
    def validate_experiment_type(self, key, experiment_type):
        """验证实验类型"""
        if not experiment_type or len(experiment_type.strip()) == 0:
            raise ValidationError("实验类型不能为空", field="experiment_type", value=experiment_type)
        return experiment_type.strip()
    
    @validates("sample_count")
    def validate_sample_count(self, key, sample_count):
        """验证样本数量"""
        if sample_count < 0:
            raise ValidationError("样本数量不能为负数", field="sample_count", value=sample_count)
        return sample_count
    
    def __repr__(self):
        return f"<ExperimentStatistics(id={self.stat_id}, type='{self.experiment_type}', samples={self.sample_count})>"