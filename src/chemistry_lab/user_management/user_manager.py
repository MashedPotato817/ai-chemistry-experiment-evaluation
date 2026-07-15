#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户管理器

提供用户注册、登录、认证和管理的核心业务逻辑。
"""

from typing import Optional, List, Tuple
from datetime import datetime

from ..database.dao import UserDAO
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, AuthenticationError, DatabaseError
from ..utils.decorators import retry
from .models import UserProfile, LoginCredentials, RegistrationData, AuthenticationResult
from .auth import AuthenticationService

logger = get_logger(__name__)


class UserManager:
    """用户管理器"""
    
    def __init__(self):
        self.user_dao = UserDAO()
        self.auth_service = AuthenticationService()
    
    @retry(max_attempts=3, delay=0.5)
    def register_user(self, registration_data: RegistrationData) -> AuthenticationResult:
        """
        注册新用户
        
        Args:
            registration_data: 注册数据
            
        Returns:
            认证结果
        """
        try:
            # 验证注册数据
            is_valid, errors = registration_data.validate()
            if not is_valid:
                return AuthenticationResult.failure_result(
                    "; ".join(errors),
                    "VALIDATION_ERROR"
                )
            
            # 检查密码强度
            is_strong, suggestions = self.auth_service.check_password_strength(registration_data.password)
            if not is_strong:
                return AuthenticationResult.failure_result(
                    f"密码强度不足: {'; '.join(suggestions)}",
                    "WEAK_PASSWORD"
                )
            
            # 检查用户名是否已存在
            existing_user = self.user_dao.get_user_by_username(registration_data.username)
            if existing_user:
                return AuthenticationResult.failure_result(
                    f"用户名已存在: {registration_data.username}",
                    "USERNAME_EXISTS"
                )
            
            # 检查邮箱是否已存在
            if registration_data.email:
                existing_email_user = self.user_dao.get_user_by_username(registration_data.email)  # 这里应该有专门的按邮箱查询方法
                # 暂时跳过邮箱重复检查，因为DAO中没有按邮箱查询的方法
            
            # 哈希密码
            password_hash = self.auth_service.hash_password(registration_data.password)
            
            # 创建用户
            db_user = self.user_dao.create_user(
                username=registration_data.username,
                password_hash=password_hash,
                email=registration_data.email
            )
            
            # 创建用户档案
            user_profile = UserProfile.from_db_user(db_user)
            
            # 生成令牌
            access_token, refresh_token = self.auth_service.generate_tokens(user_profile)
            
            logger.info(f"用户注册成功: {registration_data.username} (ID: {db_user.user_id})")
            
            return AuthenticationResult.success_result(
                user_profile=user_profile,
                access_token=access_token,
                refresh_token=refresh_token
            )
            
        except ValidationError as e:
            logger.warning(f"用户注册验证失败: {e}")
            return AuthenticationResult.failure_result(str(e), "VALIDATION_ERROR")
        except DatabaseError as e:
            logger.error(f"用户注册数据库错误: {e}")
            return AuthenticationResult.failure_result("注册失败，请稍后重试", "DATABASE_ERROR")
        except Exception as e:
            logger.error(f"用户注册未知错误: {e}")
            return AuthenticationResult.failure_result("注册失败，请联系管理员", "UNKNOWN_ERROR")
    
    def authenticate_user(self, credentials: LoginCredentials) -> AuthenticationResult:
        """
        用户认证登录
        
        Args:
            credentials: 登录凭据
            
        Returns:
            认证结果
        """
        try:
            # 验证凭据格式
            if not credentials.validate():
                return AuthenticationResult.failure_result(
                    "用户名或密码格式无效",
                    "INVALID_CREDENTIALS"
                )
            
            # 查找用户
            db_user = self.user_dao.get_user_by_username(credentials.username)
            if not db_user:
                logger.warning(f"登录失败: 用户不存在 - {credentials.username}")
                return AuthenticationResult.failure_result(
                    "用户名或密码错误",
                    "INVALID_CREDENTIALS"
                )
            
            # 检查用户状态
            if not db_user.is_active:
                logger.warning(f"登录失败: 用户已禁用 - {credentials.username}")
                return AuthenticationResult.failure_result(
                    "账户已被禁用，请联系管理员",
                    "ACCOUNT_DISABLED"
                )
            
            # 验证密码
            if not self.auth_service.verify_password(credentials.password, db_user.password_hash):
                logger.warning(f"登录失败: 密码错误 - {credentials.username}")
                return AuthenticationResult.failure_result(
                    "用户名或密码错误",
                    "INVALID_CREDENTIALS"
                )
            
            # 更新最后登录时间
            self.user_dao.update_user(db_user.user_id, last_login=datetime.now())
            
            # 创建用户档案
            user_profile = UserProfile.from_db_user(db_user)
            user_profile.last_login = datetime.now()
            
            # 生成令牌
            access_token, refresh_token = self.auth_service.generate_tokens(user_profile)
            
            logger.info(f"用户登录成功: {credentials.username} (ID: {db_user.user_id})")
            
            return AuthenticationResult.success_result(
                user_profile=user_profile,
                access_token=access_token,
                refresh_token=refresh_token if credentials.remember_me else None
            )
            
        except DatabaseError as e:
            logger.error(f"用户认证数据库错误: {e}")
            return AuthenticationResult.failure_result("登录失败，请稍后重试", "DATABASE_ERROR")
        except Exception as e:
            logger.error(f"用户认证未知错误: {e}")
            return AuthenticationResult.failure_result("登录失败，请联系管理员", "UNKNOWN_ERROR")
    
    def get_user_profile(self, user_id: int) -> Optional[UserProfile]:
        """
        获取用户档案
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户档案或None
        """
        try:
            db_user = self.user_dao.get_user_by_id(user_id)
            if db_user:
                return UserProfile.from_db_user(db_user)
            return None
        except Exception as e:
            logger.error(f"获取用户档案失败: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[UserProfile]:
        """
        根据用户名获取用户档案
        
        Args:
            username: 用户名
            
        Returns:
            用户档案或None
        """
        try:
            db_user = self.user_dao.get_user_by_username(username)
            if db_user:
                return UserProfile.from_db_user(db_user)
            return None
        except Exception as e:
            logger.error(f"根据用户名获取用户档案失败: {e}")
            return None
    
    def update_user_profile(self, user_id: int, **kwargs) -> Optional[UserProfile]:
        """
        更新用户档案
        
        Args:
            user_id: 用户ID
            **kwargs: 要更新的字段
            
        Returns:
            更新后的用户档案或None
        """
        try:
            # 过滤掉不能直接更新的字段
            allowed_fields = {'email', 'is_active', 'is_admin'}
            update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}
            
            if not update_data:
                logger.warning(f"没有有效的更新字段: {kwargs}")
                return None
            
            updated_user = self.user_dao.update_user(user_id, **update_data)
            if updated_user:
                return UserProfile.from_db_user(updated_user)
            return None
        except Exception as e:
            logger.error(f"更新用户档案失败: {e}")
            return None
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> bool:
        """
        修改密码
        
        Args:
            user_id: 用户ID
            old_password: 旧密码
            new_password: 新密码
            
        Returns:
            是否修改成功
        """
        try:
            # 获取用户
            db_user = self.user_dao.get_user_by_id(user_id)
            if not db_user:
                logger.warning(f"修改密码失败: 用户不存在 - {user_id}")
                return False
            
            # 验证旧密码
            if not self.auth_service.verify_password(old_password, db_user.password_hash):
                logger.warning(f"修改密码失败: 旧密码错误 - {user_id}")
                return False
            
            # 检查新密码强度
            is_strong, suggestions = self.auth_service.check_password_strength(new_password)
            if not is_strong:
                logger.warning(f"修改密码失败: 新密码强度不足 - {user_id}")
                raise ValidationError(f"密码强度不足: {'; '.join(suggestions)}")
            
            # 哈希新密码
            new_password_hash = self.auth_service.hash_password(new_password)
            
            # 更新密码
            updated_user = self.user_dao.update_user(user_id, password_hash=new_password_hash)
            
            if updated_user:
                logger.info(f"密码修改成功: 用户 {user_id}")
                return True
            return False
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"修改密码失败: {e}")
            return False
    
    def deactivate_user(self, user_id: int) -> bool:
        """
        停用用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否停用成功
        """
        try:
            updated_user = self.user_dao.update_user(user_id, is_active=False)
            if updated_user:
                logger.info(f"用户停用成功: {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"停用用户失败: {e}")
            return False
    
    def activate_user(self, user_id: int) -> bool:
        """
        激活用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否激活成功
        """
        try:
            updated_user = self.user_dao.update_user(user_id, is_active=True)
            if updated_user:
                logger.info(f"用户激活成功: {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"激活用户失败: {e}")
            return False
    
    def delete_user(self, user_id: int) -> bool:
        """
        删除用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否删除成功
        """
        try:
            success = self.user_dao.delete_user(user_id)
            if success:
                logger.info(f"用户删除成功: {user_id}")
            return success
        except Exception as e:
            logger.error(f"删除用户失败: {e}")
            return False
    
    def list_users(self, page: int = 1, page_size: int = 20, active_only: bool = True) -> Tuple[List[UserProfile], int]:
        """
        获取用户列表
        
        Args:
            page: 页码
            page_size: 每页大小
            active_only: 是否只返回活跃用户
            
        Returns:
            (用户档案列表, 总数)
        """
        try:
            db_users, total = self.user_dao.list_users(page, page_size, active_only)
            user_profiles = [UserProfile.from_db_user(user) for user in db_users]
            return user_profiles, total
        except Exception as e:
            logger.error(f"获取用户列表失败: {e}")
            return [], 0
    
    def verify_access_token(self, token: str) -> Optional[UserProfile]:
        """
        验证访问令牌并返回用户档案
        
        Args:
            token: 访问令牌
            
        Returns:
            用户档案或None
        """
        try:
            user_info = self.auth_service.extract_user_from_token(token)
            if user_info:
                return self.get_user_profile(user_info['user_id'])
            return None
        except Exception as e:
            logger.error(f"验证访问令牌失败: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        刷新访问令牌
        
        Args:
            refresh_token: 刷新令牌
            
        Returns:
            新的访问令牌或None
        """
        try:
            return self.auth_service.refresh_token(refresh_token)
        except Exception as e:
            logger.error(f"刷新访问令牌失败: {e}")
            return None
    
    def is_admin(self, user_id: int) -> bool:
        """
        检查用户是否为管理员
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否为管理员
        """
        try:
            user_profile = self.get_user_profile(user_id)
            return user_profile.is_admin if user_profile else False
        except Exception as e:
            logger.error(f"检查管理员权限失败: {e}")
            return False