#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库管理器

提供数据库连接、初始化和事务管理功能。
"""

import threading
from contextlib import contextmanager
from typing import Optional, Generator, Any
from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError

from .models import Base
from ..config import config
from ..utils.logger import get_logger
from ..utils.exceptions import DatabaseError
from ..utils.decorators import retry

logger = get_logger(__name__)


class DatabaseManager:
    """数据库管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化数据库管理器"""
        if hasattr(self, '_initialized'):
            return
        
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._initialized = False
        
        # 初始化数据库
        self.initialize()
    
    def initialize(self) -> None:
        """初始化数据库连接和表结构"""
        try:
            logger.info("初始化数据库连接...")
            
            # 创建数据库引擎
            self._create_engine()
            
            # 创建会话工厂
            self._create_session_factory()
            
            # 创建数据表
            self._create_tables()
            
            # 设置事件监听器
            self._setup_event_listeners()
            
            self._initialized = True
            logger.info("数据库初始化完成")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise DatabaseError(f"数据库初始化失败: {e}")
    
    def _create_engine(self) -> None:
        """创建数据库引擎"""
        try:
            # SQLite特定配置
            connect_args = {
                "check_same_thread": False,  # 允许多线程访问
                "timeout": 30,  # 连接超时时间
            }
            
            # 检查是否是 SQLite
            is_sqlite = "sqlite" in config.database.database_url
            
            # 基础引擎参数
            engine_kwargs = {
                "echo": config.database.echo,
                "echo_pool": config.database.echo_pool,
                "connect_args": connect_args,
            }
            
            # SQLite 使用 StaticPool，不支持连接池参数
            if is_sqlite:
                engine_kwargs["poolclass"] = pool.StaticPool
            else:
                # 其他数据库使用连接池参数
                engine_kwargs.update({
                    "pool_size": config.database.pool_size,
                    "max_overflow": config.database.max_overflow,
                    "pool_timeout": config.database.pool_timeout,
                    "pool_recycle": 3600,  # 1小时回收连接
                    "pool_pre_ping": True,  # 连接前检查
                })
            
            # 创建引擎
            self._engine = create_engine(
                config.database.database_url,
                **engine_kwargs
            )
            
            logger.info(f"数据库引擎创建成功: {config.database.database_url}")
            
        except Exception as e:
            logger.error(f"创建数据库引擎失败: {e}")
            raise DatabaseError(f"创建数据库引擎失败: {e}")
    
    def _create_session_factory(self) -> None:
        """创建会话工厂"""
        if not self._engine:
            raise DatabaseError("数据库引擎未初始化")
        
        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        
        logger.debug("会话工厂创建成功")
    
    def _create_tables(self) -> None:
        """创建数据表"""
        if not self._engine:
            raise DatabaseError("数据库引擎未初始化")
        
        try:
            # 创建所有表
            Base.metadata.create_all(bind=self._engine)
            logger.info("数据表创建完成")
            
            # 记录表信息
            inspector = self._engine.dialect.get_table_names(self._engine.connect())
            logger.debug(f"已创建的表: {inspector}")
            
        except Exception as e:
            logger.error(f"创建数据表失败: {e}")
            raise DatabaseError(f"创建数据表失败: {e}")
    
    def _setup_event_listeners(self) -> None:
        """设置事件监听器"""
        if not self._engine:
            return
        
        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """设置SQLite PRAGMA"""
            if "sqlite" in config.database.database_url:
                cursor = dbapi_connection.cursor()
                # 启用外键约束
                cursor.execute("PRAGMA foreign_keys=ON")
                # 设置WAL模式提高并发性能
                cursor.execute("PRAGMA journal_mode=WAL")
                # 设置同步模式
                cursor.execute("PRAGMA synchronous=NORMAL")
                # 设置缓存大小
                cursor.execute("PRAGMA cache_size=10000")
                cursor.close()
        
        @event.listens_for(self._engine, "before_cursor_execute")
        def log_sql(conn, cursor, statement, parameters, context, executemany):
            """记录SQL执行日志"""
            if config.database.echo:
                logger.debug(f"执行SQL: {statement}")
                if parameters:
                    logger.debug(f"参数: {parameters}")
        
        logger.debug("事件监听器设置完成")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """获取数据库会话上下文管理器"""
        if not self._session_factory:
            raise DatabaseError("会话工厂未初始化")
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库会话异常: {e}")
            raise DatabaseError(f"数据库操作失败: {e}")
        finally:
            session.close()
    
    def create_session(self) -> Session:
        """创建新的数据库会话"""
        if not self._session_factory:
            raise DatabaseError("会话工厂未初始化")
        
        return self._session_factory()
    
    @retry(max_attempts=3, delay=1.0, exceptions=(OperationalError,))
    def execute_sql(self, sql: str, parameters: dict = None) -> Any:
        """执行原生SQL语句"""
        if not self._engine:
            raise DatabaseError("数据库引擎未初始化")
        
        try:
            with self._engine.connect() as connection:
                result = connection.execute(sql, parameters or {})
                connection.commit()
                return result
        except Exception as e:
            logger.error(f"执行SQL失败: {sql}, 错误: {e}")
            raise DatabaseError(f"执行SQL失败: {e}", operation="execute_sql")
    
    def check_connection(self) -> bool:
        """检查数据库连接状态"""
        try:
            if not self._engine:
                return False
            
            with self._engine.connect() as connection:
                connection.execute("SELECT 1")
                return True
        except Exception as e:
            logger.warning(f"数据库连接检查失败: {e}")
            return False
    
    def get_table_info(self, table_name: str) -> dict:
        """获取表信息"""
        if not self._engine:
            raise DatabaseError("数据库引擎未初始化")
        
        try:
            from sqlalchemy import inspect
            inspector = inspect(self._engine)
            
            return {
                "columns": inspector.get_columns(table_name),
                "indexes": inspector.get_indexes(table_name),
                "foreign_keys": inspector.get_foreign_keys(table_name),
                "primary_key": inspector.get_pk_constraint(table_name),
            }
        except Exception as e:
            logger.error(f"获取表信息失败: {table_name}, 错误: {e}")
            raise DatabaseError(f"获取表信息失败: {e}", table=table_name)
    
    def backup_database(self, backup_path: str) -> None:
        """备份数据库"""
        if "sqlite" not in config.database.database_url:
            raise DatabaseError("当前只支持SQLite数据库备份")
        
        try:
            import shutil
            from urllib.parse import urlparse
            
            # 解析数据库文件路径
            parsed_url = urlparse(config.database.database_url)
            db_path = parsed_url.path.lstrip('/')
            
            # 复制数据库文件
            shutil.copy2(db_path, backup_path)
            logger.info(f"数据库备份完成: {backup_path}")
            
        except Exception as e:
            logger.error(f"数据库备份失败: {e}")
            raise DatabaseError(f"数据库备份失败: {e}")
    
    def restore_database(self, backup_path: str) -> None:
        """恢复数据库"""
        if "sqlite" not in config.database.database_url:
            raise DatabaseError("当前只支持SQLite数据库恢复")
        
        try:
            import shutil
            from urllib.parse import urlparse
            from pathlib import Path
            
            # 检查备份文件是否存在
            if not Path(backup_path).exists():
                raise DatabaseError(f"备份文件不存在: {backup_path}")
            
            # 关闭所有连接
            if self._engine:
                self._engine.dispose()
            
            # 解析数据库文件路径
            parsed_url = urlparse(config.database.database_url)
            db_path = parsed_url.path.lstrip('/')
            
            # 恢复数据库文件
            shutil.copy2(backup_path, db_path)
            
            # 重新初始化
            self.initialize()
            
            logger.info(f"数据库恢复完成: {backup_path}")
            
        except Exception as e:
            logger.error(f"数据库恢复失败: {e}")
            raise DatabaseError(f"数据库恢复失败: {e}")
    
    def close(self) -> None:
        """关闭数据库连接"""
        try:
            if self._engine:
                self._engine.dispose()
                logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")
    
    def __del__(self):
        """析构函数"""
        self.close()


# 全局数据库管理器实例
db_manager = DatabaseManager()