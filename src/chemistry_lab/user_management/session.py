#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
会话管理

提供用户会话持久化、"记住密码"功能和会话状态管理。
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import threading

from ..config import config
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, AuthenticationError
from ..utils.helpers import ensure_dir, load_json, save_json, generate_id
from .models import UserProfile, LoginCredentials
from .auth import AuthenticationService

logger = get_logger(__name__)


@dataclass
class SessionData:
    """会话数据"""
    
    session_id: str
    user_id: int
    username: str
    access_token: str
    refresh_token: Optional[str] = None
    created_at: datetime = None
    last_accessed: datetime = None
    expires_at: datetime = None
    remember_me: bool = False
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
        if self.expires_at is None:
            # 默认24小时过期，记住密码则30天
            hours = 24 * 30 if self.remember_me else 24
            self.expires_at = datetime.now() + timedelta(hours=hours)
    
    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """检查会话是否有效"""
        return not self.is_expired()
    
    def update_access_time(self):
        """更新最后访问时间"""
        self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        # 转换datetime为ISO格式字符串
        for key in ['created_at', 'last_accessed', 'expires_at']:
            if data[key]:
                data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """从字典创建会话数据"""
        # 转换ISO格式字符串为datetime
        for key in ['created_at', 'last_accessed', 'expires_at']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


class CredentialStorage:
    """凭据存储管理器"""
    
    def __init__(self):
        self.storage_dir = config.data_dir / "credentials"
        self.storage_file = self.storage_dir / "saved_credentials.json"
        ensure_dir(self.storage_dir)
        self._lock = threading.Lock()
    
    def save_credentials(self, username: str, encrypted_password: str, remember_token: str) -> None:
        """
        保存用户凭据
        
        Args:
            username: 用户名
            encrypted_password: 加密后的密码
            remember_token: 记住密码令牌
        """
        try:
            with self._lock:
                # 加载现有凭据
                credentials = self._load_credentials_file()
                
                # 添加或更新凭据
                credentials[username] = {
                    'encrypted_password': encrypted_password,
                    'remember_token': remember_token,
                    'saved_at': datetime.now().isoformat(),
                    'last_used': datetime.now().isoformat()
                }
                
                # 保存到文件
                save_json(credentials, self.storage_file)
                logger.info(f"凭据保存成功: {username}")
                
        except Exception as e:
            logger.error(f"保存凭据失败: {e}")
            raise ValidationError(f"保存凭据失败: {e}")
    
    def load_credentials(self, username: str) -> Optional[Dict[str, str]]:
        """
        加载用户凭据
        
        Args:
            username: 用户名
            
        Returns:
            凭据字典或None
        """
        try:
            with self._lock:
                credentials = self._load_credentials_file()
                user_creds = credentials.get(username)
                
                if user_creds:
                    # 更新最后使用时间
                    user_creds['last_used'] = datetime.now().isoformat()
                    save_json(credentials, self.storage_file)
                    
                    logger.debug(f"凭据加载成功: {username}")
                    return user_creds
                
                return None
                
        except Exception as e:
            logger.error(f"加载凭据失败: {e}")
            return None
    
    def remove_credentials(self, username: str) -> bool:
        """
        删除用户凭据
        
        Args:
            username: 用户名
            
        Returns:
            是否删除成功
        """
        try:
            with self._lock:
                credentials = self._load_credentials_file()
                
                if username in credentials:
                    del credentials[username]
                    save_json(credentials, self.storage_file)
                    logger.info(f"凭据删除成功: {username}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"删除凭据失败: {e}")
            return False
    
    def list_saved_usernames(self) -> list[str]:
        """
        获取所有保存的用户名
        
        Returns:
            用户名列表
        """
        try:
            credentials = self._load_credentials_file()
            return list(credentials.keys())
        except Exception as e:
            logger.error(f"获取保存的用户名失败: {e}")
            return []
    
    def cleanup_expired_credentials(self, days: int = 90) -> int:
        """
        清理过期的凭据
        
        Args:
            days: 过期天数
            
        Returns:
            清理的凭据数量
        """
        try:
            with self._lock:
                credentials = self._load_credentials_file()
                cutoff_date = datetime.now() - timedelta(days=days)
                
                expired_users = []
                for username, creds in credentials.items():
                    last_used = datetime.fromisoformat(creds.get('last_used', creds.get('saved_at')))
                    if last_used < cutoff_date:
                        expired_users.append(username)
                
                # 删除过期凭据
                for username in expired_users:
                    del credentials[username]
                
                if expired_users:
                    save_json(credentials, self.storage_file)
                    logger.info(f"清理过期凭据: {len(expired_users)} 个")
                
                return len(expired_users)
                
        except Exception as e:
            logger.error(f"清理过期凭据失败: {e}")
            return 0
    
    def _load_credentials_file(self) -> Dict[str, Any]:
        """加载凭据文件"""
        if self.storage_file.exists():
            return load_json(self.storage_file)
        return {}


class SessionManager:
    """会话管理器"""
    
    def __init__(self):
        self.auth_service = AuthenticationService()
        self.credential_storage = CredentialStorage()
        self.sessions_dir = config.data_dir / "sessions"
        ensure_dir(self.sessions_dir)
        self._active_sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
    
    def create_session(self, user_profile: UserProfile, remember_me: bool = False, 
                      user_agent: str = None, ip_address: str = None) -> SessionData:
        """
        创建新会话
        
        Args:
            user_profile: 用户档案
            remember_me: 是否记住密码
            user_agent: 用户代理
            ip_address: IP地址
            
        Returns:
            会话数据
        """
        try:
            # 生成令牌
            access_token, refresh_token = self.auth_service.generate_tokens(user_profile)
            
            # 创建会话数据
            session_data = SessionData(
                session_id=generate_id("session"),
                user_id=user_profile.user_id,
                username=user_profile.username,
                access_token=access_token,
                refresh_token=refresh_token if remember_me else None,
                remember_me=remember_me,
                user_agent=user_agent,
                ip_address=ip_address
            )
            
            # 保存会话
            self._save_session(session_data)
            
            # 如果记住密码，保存凭据
            if remember_me and refresh_token:
                self._save_remember_me_credentials(user_profile.username, refresh_token)
            
            logger.info(f"会话创建成功: {session_data.session_id} (用户: {user_profile.username})")
            return session_data
            
        except Exception as e:
            logger.error(f"创建会话失败: {e}")
            raise AuthenticationError(f"创建会话失败: {e}")
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        获取会话数据
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话数据或None
        """
        try:
            with self._lock:
                # 先从内存中查找
                if session_id in self._active_sessions:
                    session = self._active_sessions[session_id]
                    if session.is_valid():
                        session.update_access_time()
                        return session
                    else:
                        # 会话过期，从内存中删除
                        del self._active_sessions[session_id]
                
                # 从文件中加载
                session = self._load_session(session_id)
                if session and session.is_valid():
                    session.update_access_time()
                    self._active_sessions[session_id] = session
                    return session
                elif session:
                    # 会话过期，删除文件
                    self._delete_session_file(session_id)
                
                return None
                
        except Exception as e:
            logger.error(f"获取会话失败: {e}")
            return None
    
    def refresh_session(self, session_id: str) -> Optional[SessionData]:
        """
        刷新会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            刷新后的会话数据或None
        """
        try:
            session = self.get_session(session_id)
            if not session or not session.refresh_token:
                return None
            
            # 使用刷新令牌生成新的访问令牌
            new_access_token = self.auth_service.refresh_token(session.refresh_token)
            if not new_access_token:
                return None
            
            # 更新会话
            session.access_token = new_access_token
            session.update_access_time()
            
            # 保存更新后的会话
            self._save_session(session)
            
            logger.info(f"会话刷新成功: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"刷新会话失败: {e}")
            return None
    
    def destroy_session(self, session_id: str) -> bool:
        """
        销毁会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否销毁成功
        """
        try:
            with self._lock:
                # 从内存中删除
                if session_id in self._active_sessions:
                    del self._active_sessions[session_id]
                
                # 删除文件
                success = self._delete_session_file(session_id)
                
                if success:
                    logger.info(f"会话销毁成功: {session_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"销毁会话失败: {e}")
            return False
    
    def get_saved_credentials(self, username: str) -> Optional[str]:
        """
        获取保存的凭据令牌
        
        Args:
            username: 用户名
            
        Returns:
            刷新令牌或None
        """
        try:
            credentials = self.credential_storage.load_credentials(username)
            if credentials:
                return credentials.get('remember_token')
            return None
        except Exception as e:
            logger.error(f"获取保存的凭据失败: {e}")
            return None
    
    def auto_login_with_saved_credentials(self, username: str) -> Optional[SessionData]:
        """
        使用保存的凭据自动登录
        
        Args:
            username: 用户名
            
        Returns:
            会话数据或None
        """
        try:
            # 获取保存的刷新令牌
            refresh_token = self.get_saved_credentials(username)
            if not refresh_token:
                return None
            
            # 验证刷新令牌
            payload = self.auth_service.jwt_manager.verify_token(refresh_token)
            if not payload or payload.get('type') != 'refresh':
                # 令牌无效，删除保存的凭据
                self.credential_storage.remove_credentials(username)
                return None
            
            # 生成新的访问令牌
            new_access_token = self.auth_service.refresh_token(refresh_token)
            if not new_access_token:
                # 刷新失败，删除保存的凭据
                self.credential_storage.remove_credentials(username)
                return None
            
            # 创建会话数据
            session_data = SessionData(
                session_id=generate_id("session"),
                user_id=payload['user_id'],
                username=payload['username'],
                access_token=new_access_token,
                refresh_token=refresh_token,
                remember_me=True
            )
            
            # 保存会话
            self._save_session(session_data)
            
            logger.info(f"自动登录成功: {username}")
            return session_data
            
        except Exception as e:
            logger.error(f"自动登录失败: {e}")
            return None
    
    def remove_saved_credentials(self, username: str) -> bool:
        """
        删除保存的凭据
        
        Args:
            username: 用户名
            
        Returns:
            是否删除成功
        """
        return self.credential_storage.remove_credentials(username)
    
    def list_saved_usernames(self) -> list[str]:
        """获取所有保存凭据的用户名"""
        return self.credential_storage.list_saved_usernames()
    
    def cleanup_expired_sessions(self) -> int:
        """
        清理过期会话
        
        Returns:
            清理的会话数量
        """
        try:
            cleaned_count = 0
            
            # 清理内存中的过期会话
            with self._lock:
                expired_sessions = [
                    session_id for session_id, session in self._active_sessions.items()
                    if session.is_expired()
                ]
                
                for session_id in expired_sessions:
                    del self._active_sessions[session_id]
                    cleaned_count += 1
            
            # 清理文件中的过期会话
            for session_file in self.sessions_dir.glob("session_*.json"):
                try:
                    session_data = load_json(session_file)
                    session = SessionData.from_dict(session_data)
                    if session.is_expired():
                        session_file.unlink()
                        cleaned_count += 1
                except Exception:
                    # 文件损坏，直接删除
                    session_file.unlink()
                    cleaned_count += 1
            
            # 清理过期的保存凭据
            expired_creds = self.credential_storage.cleanup_expired_credentials()
            
            if cleaned_count > 0 or expired_creds > 0:
                logger.info(f"清理完成: {cleaned_count} 个过期会话, {expired_creds} 个过期凭据")
            
            return cleaned_count + expired_creds
            
        except Exception as e:
            logger.error(f"清理过期会话失败: {e}")
            return 0
    
    def _save_session(self, session_data: SessionData) -> None:
        """保存会话到文件"""
        try:
            session_file = self.sessions_dir / f"{session_data.session_id}.json"
            save_json(session_data.to_dict(), session_file)
            
            # 同时保存到内存
            with self._lock:
                self._active_sessions[session_data.session_id] = session_data
                
        except Exception as e:
            logger.error(f"保存会话文件失败: {e}")
            raise
    
    def _load_session(self, session_id: str) -> Optional[SessionData]:
        """从文件加载会话"""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_data = load_json(session_file)
                return SessionData.from_dict(session_data)
            return None
        except Exception as e:
            logger.error(f"加载会话文件失败: {e}")
            return None
    
    def _delete_session_file(self, session_id: str) -> bool:
        """删除会话文件"""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"删除会话文件失败: {e}")
            return False
    
    def _save_remember_me_credentials(self, username: str, refresh_token: str) -> None:
        """保存记住密码的凭据"""
        try:
            # 这里我们直接保存刷新令牌，实际应用中可能需要额外加密
            self.credential_storage.save_credentials(
                username=username,
                encrypted_password="",  # 不保存密码，只保存令牌
                remember_token=refresh_token
            )
        except Exception as e:
            logger.error(f"保存记住密码凭据失败: {e}")
            # 不抛出异常，因为这不是关键功能