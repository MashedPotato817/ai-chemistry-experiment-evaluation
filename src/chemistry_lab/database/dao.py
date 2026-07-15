#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据访问对象(DAO)

提供数据库CRUD操作和查询方法的封装。
"""

import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy import and_, or_, desc, asc, func

from .models import User, ExperimentSession, ExperimentStep, InteractionLog, ExperimentStatistics
from .database import db_manager
from ..utils.logger import get_logger
from ..utils.exceptions import DatabaseError, ValidationError
from ..utils.decorators import retry

logger = get_logger(__name__)


class BaseDAO:
    """基础DAO类"""
    
    def __init__(self):
        self.db_manager = db_manager
    
    def _handle_db_error(self, e: Exception, operation: str, **kwargs) -> None:
        """处理数据库错误"""
        if isinstance(e, IntegrityError):
            logger.error(f"数据完整性错误 - {operation}: {e}")
            raise DatabaseError(f"数据完整性错误: {e}", operation=operation, **kwargs)
        elif isinstance(e, NoResultFound):
            logger.warning(f"未找到数据 - {operation}: {e}")
            raise DatabaseError(f"未找到数据: {e}", operation=operation, **kwargs)
        else:
            logger.error(f"数据库操作失败 - {operation}: {e}")
            raise DatabaseError(f"数据库操作失败: {e}", operation=operation, **kwargs)


class UserDAO(BaseDAO):
    """用户数据访问对象"""
    
    @retry(max_attempts=3, delay=0.5)
    def create_user(self, username: str, password_hash: str, email: str = None) -> User:
        """
        创建新用户
        
        Args:
            username: 用户名
            password_hash: 密码哈希
            email: 邮箱地址
            
        Returns:
            创建的用户对象
            
        Raises:
            DatabaseError: 数据库操作失败
            ValidationError: 数据验证失败
        """
        try:
            with self.db_manager.get_session() as session:
                # 检查用户名是否已存在
                existing_user = session.query(User).filter_by(username=username).first()
                if existing_user:
                    raise ValidationError(f"用户名已存在: {username}", field="username", value=username)
                
                # 检查邮箱是否已存在
                if email:
                    existing_email = session.query(User).filter_by(email=email).first()
                    if existing_email:
                        raise ValidationError(f"邮箱已存在: {email}", field="email", value=email)
                
                # 创建用户
                user = User(
                    username=username,
                    password_hash=password_hash,
                    email=email
                )
                
                session.add(user)
                session.commit()
                
                logger.info(f"用户创建成功: {username} (ID: {user.user_id})")
                return user
                
        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            self._handle_db_error(e, "create_user", username=username)
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        根据ID获取用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户对象或None
        """
        try:
            with self.db_manager.get_session() as session:
                user = session.query(User).filter_by(user_id=user_id).first()
                return user
        except Exception as e:
            self._handle_db_error(e, "get_user_by_id", user_id=user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        根据用户名获取用户
        
        Args:
            username: 用户名
            
        Returns:
            用户对象或None
        """
        try:
            with self.db_manager.get_session() as session:
                user = session.query(User).filter_by(username=username).first()
                return user
        except Exception as e:
            self._handle_db_error(e, "get_user_by_username", username=username)
    
    def authenticate_user(self, username: str, password_hash: str) -> Optional[User]:
        """
        验证用户凭据
        
        Args:
            username: 用户名
            password_hash: 密码哈希
            
        Returns:
            验证成功的用户对象或None
        """
        try:
            with self.db_manager.get_session() as session:
                user = session.query(User).filter(
                    and_(
                        User.username == username,
                        User.password_hash == password_hash,
                        User.is_active == True
                    )
                ).first()
                
                if user:
                    # 更新最后登录时间
                    user.last_login = datetime.now()
                    session.commit()
                    logger.info(f"用户认证成功: {username}")
                
                return user
        except Exception as e:
            self._handle_db_error(e, "authenticate_user", username=username)
    
    def update_user(self, user_id: int, **kwargs) -> Optional[User]:
        """
        更新用户信息
        
        Args:
            user_id: 用户ID
            **kwargs: 要更新的字段
            
        Returns:
            更新后的用户对象或None
        """
        try:
            with self.db_manager.get_session() as session:
                user = session.query(User).filter_by(user_id=user_id).first()
                if not user:
                    return None
                
                # 更新字段
                for key, value in kwargs.items():
                    if hasattr(user, key):
                        setattr(user, key, value)
                
                session.commit()
                logger.info(f"用户更新成功: {user_id}")
                return user
        except Exception as e:
            self._handle_db_error(e, "update_user", user_id=user_id)
    
    def delete_user(self, user_id: int) -> bool:
        """
        删除用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            删除是否成功
        """
        try:
            with self.db_manager.get_session() as session:
                user = session.query(User).filter_by(user_id=user_id).first()
                if not user:
                    return False
                
                session.delete(user)
                session.commit()
                logger.info(f"用户删除成功: {user_id}")
                return True
        except Exception as e:
            self._handle_db_error(e, "delete_user", user_id=user_id)
    
    def list_users(self, page: int = 1, page_size: int = 20, active_only: bool = True) -> Tuple[List[User], int]:
        """
        获取用户列表
        
        Args:
            page: 页码
            page_size: 每页大小
            active_only: 是否只返回活跃用户
            
        Returns:
            (用户列表, 总数)
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(User)
                
                if active_only:
                    query = query.filter(User.is_active == True)
                
                total = query.count()
                
                users = query.order_by(desc(User.created_at)).offset(
                    (page - 1) * page_size
                ).limit(page_size).all()
                
                return users, total
        except Exception as e:
            self._handle_db_error(e, "list_users")


class ExperimentDAO(BaseDAO):
    """实验数据访问对象"""
    
    @retry(max_attempts=3, delay=0.5)
    def create_experiment_session(self, user_id: int, experiment_type: str, **kwargs) -> ExperimentSession:
        """
        创建实验会话
        
        Args:
            user_id: 用户ID
            experiment_type: 实验类型
            **kwargs: 其他参数
            
        Returns:
            创建的实验会话对象
        """
        try:
            with self.db_manager.get_session() as session:
                # 验证用户存在
                user = session.query(User).filter_by(user_id=user_id).first()
                if not user:
                    raise ValidationError(f"用户不存在: {user_id}", field="user_id", value=user_id)
                
                experiment_session = ExperimentSession(
                    user_id=user_id,
                    experiment_type=experiment_type,
                    start_time=kwargs.get('start_time', datetime.now()),
                    **{k: v for k, v in kwargs.items() if k != 'start_time'}
                )
                
                session.add(experiment_session)
                session.commit()
                
                logger.info(f"实验会话创建成功: {experiment_session.session_id}")
                return experiment_session
                
        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            self._handle_db_error(e, "create_experiment_session", user_id=user_id, experiment_type=experiment_type)
    
    def get_experiment_session(self, session_id: int) -> Optional[ExperimentSession]:
        """
        获取实验会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            实验会话对象或None
        """
        try:
            with self.db_manager.get_session() as session:
                experiment_session = session.query(ExperimentSession).filter_by(session_id=session_id).first()
                return experiment_session
        except Exception as e:
            self._handle_db_error(e, "get_experiment_session", session_id=session_id)
    
    def update_experiment_session(self, session_id: int, **kwargs) -> Optional[ExperimentSession]:
        """
        更新实验会话
        
        Args:
            session_id: 会话ID
            **kwargs: 要更新的字段
            
        Returns:
            更新后的实验会话对象或None
        """
        try:
            with self.db_manager.get_session() as session:
                experiment_session = session.query(ExperimentSession).filter_by(session_id=session_id).first()
                if not experiment_session:
                    return None
                
                for key, value in kwargs.items():
                    if hasattr(experiment_session, key):
                        setattr(experiment_session, key, value)
                
                session.commit()
                logger.info(f"实验会话更新成功: {session_id}")
                return experiment_session
        except Exception as e:
            self._handle_db_error(e, "update_experiment_session", session_id=session_id)
    
    def complete_experiment_session(self, session_id: int, final_score: float, s1_score: float, s2_score: float) -> Optional[ExperimentSession]:
        """
        完成实验会话
        
        Args:
            session_id: 会话ID
            final_score: 最终得分
            s1_score: S1得分
            s2_score: S2得分
            
        Returns:
            更新后的实验会话对象或None
        """
        try:
            with self.db_manager.get_session() as session:
                experiment_session = session.query(ExperimentSession).filter_by(session_id=session_id).first()
                if not experiment_session:
                    return None
                
                experiment_session.end_time = datetime.now()
                experiment_session.total_duration = experiment_session.calculate_duration()
                experiment_session.final_score = final_score
                experiment_session.s1_score = s1_score
                experiment_session.s2_score = s2_score
                experiment_session.status = "completed"
                
                session.commit()
                logger.info(f"实验会话完成: {session_id}, 得分: {final_score}")
                return experiment_session
        except Exception as e:
            self._handle_db_error(e, "complete_experiment_session", session_id=session_id)
    
    def add_experiment_step(self, session_id: int, step_name: str, start_timestamp: float, 
                           end_timestamp: float, sequence_order: int, **kwargs) -> ExperimentStep:
        """
        添加实验步骤
        
        Args:
            session_id: 会话ID
            step_name: 步骤名称
            start_timestamp: 开始时间戳
            end_timestamp: 结束时间戳
            sequence_order: 步骤顺序
            **kwargs: 其他参数
            
        Returns:
            创建的实验步骤对象
        """
        try:
            with self.db_manager.get_session() as session:
                # 验证会话存在
                experiment_session = session.query(ExperimentSession).filter_by(session_id=session_id).first()
                if not experiment_session:
                    raise ValidationError(f"实验会话不存在: {session_id}", field="session_id", value=session_id)
                
                duration = end_timestamp - start_timestamp
                
                step = ExperimentStep(
                    session_id=session_id,
                    step_name=step_name,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    duration=duration,
                    sequence_order=sequence_order,
                    **kwargs
                )
                
                session.add(step)
                session.commit()
                
                logger.info(f"实验步骤添加成功: {step.step_id}")
                return step
                
        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            self._handle_db_error(e, "add_experiment_step", session_id=session_id, step_name=step_name)
    
    def get_experiment_steps(self, session_id: int) -> List[ExperimentStep]:
        """
        获取实验步骤列表
        
        Args:
            session_id: 会话ID
            
        Returns:
            实验步骤列表
        """
        try:
            with self.db_manager.get_session() as session:
                steps = session.query(ExperimentStep).filter_by(
                    session_id=session_id
                ).order_by(ExperimentStep.sequence_order).all()
                return steps
        except Exception as e:
            self._handle_db_error(e, "get_experiment_steps", session_id=session_id)
    
    def add_interaction_log(self, session_id: int, timestamp: float, interaction_type: str, 
                           user_input: str = None, system_response: str = None, **kwargs) -> InteractionLog:
        """
        添加交互日志
        
        Args:
            session_id: 会话ID
            timestamp: 时间戳
            interaction_type: 交互类型
            user_input: 用户输入
            system_response: 系统响应
            **kwargs: 其他参数
            
        Returns:
            创建的交互日志对象
        """
        try:
            with self.db_manager.get_session() as session:
                log = InteractionLog(
                    session_id=session_id,
                    timestamp=timestamp,
                    interaction_type=interaction_type,
                    user_input=user_input,
                    system_response=system_response,
                    **kwargs
                )
                
                session.add(log)
                session.commit()
                
                logger.debug(f"交互日志添加成功: {log.log_id}")
                return log
                
        except Exception as e:
            self._handle_db_error(e, "add_interaction_log", session_id=session_id, interaction_type=interaction_type)
    
    def get_user_experiments(self, user_id: int, page: int = 1, page_size: int = 20) -> Tuple[List[ExperimentSession], int]:
        """
        获取用户的实验记录
        
        Args:
            user_id: 用户ID
            page: 页码
            page_size: 每页大小
            
        Returns:
            (实验会话列表, 总数)
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(ExperimentSession).filter_by(user_id=user_id)
                total = query.count()
                
                experiments = query.order_by(desc(ExperimentSession.start_time)).offset(
                    (page - 1) * page_size
                ).limit(page_size).all()
                
                return experiments, total
        except Exception as e:
            self._handle_db_error(e, "get_user_experiments", user_id=user_id)
    
    def get_experiments_by_type(self, experiment_type: str, limit: int = 100) -> List[ExperimentSession]:
        """
        根据实验类型获取实验记录
        
        Args:
            experiment_type: 实验类型
            limit: 限制数量
            
        Returns:
            实验会话列表
        """
        try:
            with self.db_manager.get_session() as session:
                experiments = session.query(ExperimentSession).filter(
                    and_(
                        ExperimentSession.experiment_type == experiment_type,
                        ExperimentSession.status == "completed"
                    )
                ).order_by(desc(ExperimentSession.start_time)).limit(limit).all()
                
                return experiments
        except Exception as e:
            self._handle_db_error(e, "get_experiments_by_type", experiment_type=experiment_type)


class StatisticsDAO(BaseDAO):
    """统计数据访问对象"""
    
    def get_or_create_statistics(self, experiment_type: str) -> ExperimentStatistics:
        """
        获取或创建实验统计记录
        
        Args:
            experiment_type: 实验类型
            
        Returns:
            实验统计对象
        """
        try:
            with self.db_manager.get_session() as session:
                stats = session.query(ExperimentStatistics).filter_by(experiment_type=experiment_type).first()
                
                if not stats:
                    stats = ExperimentStatistics(
                        experiment_type=experiment_type,
                        mean_duration=0.0,
                        std_deviation=0.0,
                        sample_count=0
                    )
                    session.add(stats)
                    session.commit()
                    logger.info(f"创建新的统计记录: {experiment_type}")
                
                return stats
        except Exception as e:
            self._handle_db_error(e, "get_or_create_statistics", experiment_type=experiment_type)
    
    def update_statistics(self, experiment_type: str, mean_duration: float, std_deviation: float, 
                         sample_count: int, **kwargs) -> ExperimentStatistics:
        """
        更新实验统计数据
        
        Args:
            experiment_type: 实验类型
            mean_duration: 平均持续时间
            std_deviation: 标准差
            sample_count: 样本数量
            **kwargs: 其他参数
            
        Returns:
            更新后的统计对象
        """
        try:
            with self.db_manager.get_session() as session:
                stats = session.query(ExperimentStatistics).filter_by(experiment_type=experiment_type).first()
                
                if not stats:
                    stats = ExperimentStatistics(experiment_type=experiment_type)
                    session.add(stats)
                
                stats.mean_duration = mean_duration
                stats.std_deviation = std_deviation
                stats.sample_count = sample_count
                
                for key, value in kwargs.items():
                    if hasattr(stats, key):
                        setattr(stats, key, value)
                
                session.commit()
                logger.info(f"统计数据更新成功: {experiment_type}")
                return stats
        except Exception as e:
            self._handle_db_error(e, "update_statistics", experiment_type=experiment_type)
    
    def calculate_experiment_statistics(self, experiment_type: str) -> Dict[str, Any]:
        """
        计算实验统计数据
        
        Args:
            experiment_type: 实验类型
            
        Returns:
            统计数据字典
        """
        try:
            with self.db_manager.get_session() as session:
                # 获取已完成的实验数据
                experiments = session.query(ExperimentSession).filter(
                    and_(
                        ExperimentSession.experiment_type == experiment_type,
                        ExperimentSession.status == "completed",
                        ExperimentSession.total_duration.isnot(None)
                    )
                ).all()
                
                if not experiments:
                    return {
                        "mean_duration": 0.0,
                        "std_deviation": 0.0,
                        "sample_count": 0,
                        "mean_score": 0.0,
                        "score_std_deviation": 0.0
                    }
                
                # 计算持续时间统计
                durations = [exp.total_duration for exp in experiments]
                scores = [exp.final_score for exp in experiments if exp.final_score is not None]
                
                import statistics
                
                mean_duration = statistics.mean(durations)
                std_deviation = statistics.stdev(durations) if len(durations) > 1 else 0.0
                
                mean_score = statistics.mean(scores) if scores else 0.0
                score_std_deviation = statistics.stdev(scores) if len(scores) > 1 else 0.0
                
                return {
                    "mean_duration": mean_duration,
                    "std_deviation": std_deviation,
                    "sample_count": len(experiments),
                    "mean_score": mean_score,
                    "score_std_deviation": score_std_deviation
                }
                
        except Exception as e:
            self._handle_db_error(e, "calculate_experiment_statistics", experiment_type=experiment_type)
    
    def get_all_statistics(self) -> List[ExperimentStatistics]:
        """
        获取所有实验统计数据
        
        Returns:
            统计数据列表
        """
        try:
            with self.db_manager.get_session() as session:
                stats = session.query(ExperimentStatistics).order_by(
                    desc(ExperimentStatistics.last_updated)
                ).all()
                return stats
        except Exception as e:
            self._handle_db_error(e, "get_all_statistics")



class ExperimentStatisticsDAO(BaseDAO):
    """实验统计数据访问对象"""
    
    @retry(max_attempts=3, delay=0.5)
    def create_statistics(
        self,
        experiment_type: str,
        mean_duration: float,
        std_deviation: float,
        sample_count: int
    ) -> ExperimentStatistics:
        """
        创建实验统计记录
        
        Args:
            experiment_type: 实验类型
            mean_duration: 平均时长
            std_deviation: 标准差
            sample_count: 样本数量
        
        Returns:
            ExperimentStatistics对象
        """
        try:
            with self.db_manager.get_session() as session:
                stats = ExperimentStatistics(
                    experiment_type=experiment_type,
                    mean_duration=mean_duration,
                    std_deviation=std_deviation,
                    sample_count=sample_count
                )
                session.add(stats)
                session.commit()
                session.refresh(stats)
                
                logger.info(f"创建实验统计记录: {experiment_type}")
                return stats
                
        except Exception as e:
            self._handle_db_error(e, "create_statistics", experiment_type=experiment_type)
    
    @retry(max_attempts=3, delay=0.5)
    def get_statistics(self, experiment_type: str) -> Optional[Dict[str, Any]]:
        """
        获取实验统计参数
        
        Args:
            experiment_type: 实验类型
        
        Returns:
            统计参数字典或None
        """
        try:
            with self.db_manager.get_session() as session:
                stats = session.query(ExperimentStatistics).filter(
                    ExperimentStatistics.experiment_type == experiment_type
                ).first()
                
                if stats:
                    return {
                        'experiment_type': stats.experiment_type,
                        'mean_duration': stats.mean_duration,
                        'std_deviation': stats.std_deviation,
                        'sample_count': stats.sample_count,
                        'last_updated': stats.last_updated
                    }
                return None
                
        except Exception as e:
            self._handle_db_error(e, "get_statistics", experiment_type=experiment_type)
    
    @retry(max_attempts=3, delay=0.5)
    def update_statistics(
        self,
        experiment_type: str,
        mean_duration: float,
        std_deviation: float,
        sample_count: int
    ) -> bool:
        """
        更新实验统计参数
        
        Args:
            experiment_type: 实验类型
            mean_duration: 新的平均时长
            std_deviation: 新的标准差
            sample_count: 新的样本数量
        
        Returns:
            是否更新成功
        """
        try:
            with self.db_manager.get_session() as session:
                stats = session.query(ExperimentStatistics).filter(
                    ExperimentStatistics.experiment_type == experiment_type
                ).first()
                
                if stats:
                    stats.mean_duration = mean_duration
                    stats.std_deviation = std_deviation
                    stats.sample_count = sample_count
                    stats.last_updated = datetime.now()
                    session.commit()
                    
                    logger.info(f"更新实验统计: {experiment_type}")
                    return True
                else:
                    logger.warning(f"未找到实验统计记录: {experiment_type}")
                    return False
                    
        except Exception as e:
            self._handle_db_error(e, "update_statistics", experiment_type=experiment_type)
    
    @retry(max_attempts=3, delay=0.5)
    def get_all_statistics(self) -> List[Dict[str, Any]]:
        """
        获取所有实验统计
        
        Returns:
            统计参数列表
        """
        try:
            with self.db_manager.get_session() as session:
                stats_list = session.query(ExperimentStatistics).all()
                
                return [
                    {
                        'experiment_type': stats.experiment_type,
                        'mean_duration': stats.mean_duration,
                        'std_deviation': stats.std_deviation,
                        'sample_count': stats.sample_count,
                        'last_updated': stats.last_updated
                    }
                    for stats in stats_list
                ]
                
        except Exception as e:
            self._handle_db_error(e, "get_all_statistics")
