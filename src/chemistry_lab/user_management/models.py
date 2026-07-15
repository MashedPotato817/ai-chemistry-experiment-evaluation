#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户管理数据模型

定义用户相关的数据模型和业务逻辑。
"""

from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# 重新导出数据库模型中的User类
from ..database.models import User as DBUser


@dataclass
class UserProfile:
    """用户档案信息"""
    
    user_id: int
    username: str
    email: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    
    # 扩展信息
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_db_user(cls, db_user: DBUser) -> 'UserProfile':
        """从数据库用户对象创建用户档案"""
        return cls(
            user_id=db_user.user_id,
            username=db_user.username,
            email=db_user.email,
            is_active=db_user.is_active,
            is_admin=db_user.is_admin,
            created_at=db_user.created_at,
            last_login=db_user.last_login,
            display_name=db_user.username,  # 默认使用用户名作为显示名
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "display_name": self.display_name,
            "avatar_url": self.avatar_url,
            "preferences": self.preferences,
        }
    
    def is_authenticated(self) -> bool:
        """检查用户是否已认证"""
        return self.is_active and self.user_id > 0


@dataclass
class LoginCredentials:
    """登录凭据"""
    
    username: str
    password: str
    remember_me: bool = False
    
    def validate(self) -> bool:
        """验证凭据格式"""
        return (
            self.username and len(self.username.strip()) >= 3 and
            self.password and len(self.password) >= 6
        )


@dataclass
class RegistrationData:
    """注册数据"""
    
    username: str
    password: str
    email: Optional[str] = None
    confirm_password: Optional[str] = None
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        验证注册数据
        
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        # 验证用户名
        if not self.username or len(self.username.strip()) < 3:
            errors.append("用户名长度至少3个字符")
        elif len(self.username) > 50:
            errors.append("用户名长度不能超过50个字符")
        elif not self.username.replace('_', '').replace('-', '').isalnum():
            errors.append("用户名只能包含字母、数字、下划线和连字符")
        
        # 验证密码
        if not self.password:
            errors.append("密码不能为空")
        elif len(self.password) > 128:
            errors.append("密码长度不能超过128个字符")
        
        # 验证确认密码
        if self.confirm_password is not None and self.password != self.confirm_password:
            errors.append("两次输入的密码不一致")
        
        # 验证邮箱
        if self.email:
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, self.email):
                errors.append("邮箱地址格式无效")
        
        return len(errors) == 0, errors


@dataclass
class AuthenticationResult:
    """认证结果"""
    
    success: bool
    user_profile: Optional[UserProfile] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    @classmethod
    def success_result(cls, user_profile: UserProfile, access_token: str, 
                      refresh_token: str = None) -> 'AuthenticationResult':
        """创建成功的认证结果"""
        return cls(
            success=True,
            user_profile=user_profile,
            access_token=access_token,
            refresh_token=refresh_token
        )
    
    @classmethod
    def failure_result(cls, error_message: str, error_code: str = None) -> 'AuthenticationResult':
        """创建失败的认证结果"""
        return cls(
            success=False,
            error_message=error_message,
            error_code=error_code or "AUTHENTICATION_FAILED"
        )


# 重新导出数据库User类，方便外部使用
User = DBUser
