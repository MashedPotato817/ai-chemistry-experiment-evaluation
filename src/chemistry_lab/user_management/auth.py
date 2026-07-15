#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认证服务

提供密码哈希、JWT令牌管理和用户认证功能。
"""

import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import bcrypt

from ..config import config
from ..utils.logger import get_logger
from ..utils.exceptions import AuthenticationError, ValidationError
from .models import UserProfile

logger = get_logger(__name__)


class PasswordManager:
    """密码管理器"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        哈希密码
        
        Args:
            password: 明文密码
            
        Returns:
            哈希后的密码
        """
        if not password:
            raise ValidationError("密码不能为空", field="password")
        
        # 使用bcrypt进行密码哈希
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """
        验证密码
        
        Args:
            password: 明文密码
            hashed_password: 哈希密码
            
        Returns:
            密码是否匹配
        """
        if not password or not hashed_password:
            return False
        
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception as e:
            logger.error(f"密码验证失败: {e}")
            return False
    
    @staticmethod
    def generate_random_password(length: int = 12) -> str:
        """
        生成随机密码
        
        Args:
            length: 密码长度
            
        Returns:
            随机密码
        """
        import string
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def check_password_strength(password: str) -> tuple[bool, list[str]]:
        """
        检查密码是否可用。

        项目演示环境不再要求长度、大小写、数字或特殊字符组合；
        仅拒绝空密码，确保注册和修改密码流程不会创建无密码账号。
        
        Args:
            password: 密码
            
        Returns:
            (是否可用, 错误信息列表)
        """
        if not password:
            return False, ["密码不能为空"]
        return True, []


class JWTManager:
    """JWT令牌管理器"""
    
    def __init__(self):
        self.secret_key = config.secret_key
        self.algorithm = config.jwt_algorithm
        self.expiration_hours = config.jwt_expiration_hours
    
    def generate_access_token(self, user_profile: UserProfile) -> str:
        """
        生成访问令牌
        
        Args:
            user_profile: 用户档案
            
        Returns:
            JWT访问令牌
        """
        now = datetime.utcnow()
        payload = {
            'user_id': user_profile.user_id,
            'username': user_profile.username,
            'is_admin': user_profile.is_admin,
            'iat': now,
            'exp': now + timedelta(hours=self.expiration_hours),
            'type': 'access'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"生成访问令牌: 用户 {user_profile.username}")
        return token
    
    def generate_refresh_token(self, user_profile: UserProfile) -> str:
        """
        生成刷新令牌
        
        Args:
            user_profile: 用户档案
            
        Returns:
            JWT刷新令牌
        """
        now = datetime.utcnow()
        payload = {
            'user_id': user_profile.user_id,
            'username': user_profile.username,
            'iat': now,
            'exp': now + timedelta(days=30),  # 刷新令牌有效期30天
            'type': 'refresh'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"生成刷新令牌: 用户 {user_profile.username}")
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        验证令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            令牌载荷或None
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("令牌已过期")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"无效令牌: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        使用刷新令牌生成新的访问令牌
        
        Args:
            refresh_token: 刷新令牌
            
        Returns:
            新的访问令牌或None
        """
        payload = self.verify_token(refresh_token)
        if not payload or payload.get('type') != 'refresh':
            return None
        
        # 创建新的访问令牌载荷
        now = datetime.utcnow()
        new_payload = {
            'user_id': payload['user_id'],
            'username': payload['username'],
            'is_admin': payload.get('is_admin', False),
            'iat': now,
            'exp': now + timedelta(hours=self.expiration_hours),
            'type': 'access'
        }
        
        new_token = jwt.encode(new_payload, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"刷新访问令牌: 用户 {payload['username']}")
        return new_token
    
    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """
        获取令牌信息（不验证过期时间）
        
        Args:
            token: JWT令牌
            
        Returns:
            令牌信息或None
        """
        try:
            # 不验证过期时间
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": False}
            )
            return payload
        except jwt.InvalidTokenError:
            return None


class AuthenticationService:
    """认证服务"""
    
    def __init__(self):
        self.password_manager = PasswordManager()
        self.jwt_manager = JWTManager()
    
    def hash_password(self, password: str) -> str:
        """哈希密码"""
        return self.password_manager.hash_password(password)
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """验证密码"""
        return self.password_manager.verify_password(password, hashed_password)
    
    def generate_tokens(self, user_profile: UserProfile) -> tuple[str, str]:
        """
        生成访问令牌和刷新令牌
        
        Args:
            user_profile: 用户档案
            
        Returns:
            (访问令牌, 刷新令牌)
        """
        access_token = self.jwt_manager.generate_access_token(user_profile)
        refresh_token = self.jwt_manager.generate_refresh_token(user_profile)
        return access_token, refresh_token
    
    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        验证访问令牌
        
        Args:
            token: 访问令牌
            
        Returns:
            令牌载荷或None
        """
        payload = self.jwt_manager.verify_token(token)
        if payload and payload.get('type') == 'access':
            return payload
        return None
    
    def refresh_token(self, refresh_token: str) -> Optional[str]:
        """刷新访问令牌"""
        return self.jwt_manager.refresh_access_token(refresh_token)
    
    def validate_token_user(self, token: str, required_user_id: int = None) -> bool:
        """
        验证令牌用户
        
        Args:
            token: 访问令牌
            required_user_id: 要求的用户ID
            
        Returns:
            是否验证通过
        """
        payload = self.verify_access_token(token)
        if not payload:
            return False
        
        if required_user_id and payload.get('user_id') != required_user_id:
            return False
        
        return True
    
    def extract_user_from_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        从令牌中提取用户信息
        
        Args:
            token: 访问令牌
            
        Returns:
            用户信息字典或None
        """
        payload = self.verify_access_token(token)
        if payload:
            return {
                'user_id': payload.get('user_id'),
                'username': payload.get('username'),
                'is_admin': payload.get('is_admin', False)
            }
        return None
    
    def check_password_strength(self, password: str) -> tuple[bool, list[str]]:
        """检查密码强度"""
        return self.password_manager.check_password_strength(password)
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        生成安全令牌（用于重置密码等）
        
        Args:
            length: 令牌长度
            
        Returns:
            安全令牌
        """
        return secrets.token_urlsafe(length)
