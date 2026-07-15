#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户认证属性测试

**特征: ai-chemistry-lab-assessment, 属性 1: 用户认证一致性**
验证用户认证系统的一致性属性。
"""

import pytest
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

# 导入被测试的模块
from src.chemistry_lab.database.models import Base
from src.chemistry_lab.user_management.user_manager import UserManager
from src.chemistry_lab.user_management.models import LoginCredentials, RegistrationData, UserProfile
from src.chemistry_lab.user_management.auth import AuthenticationService, PasswordManager, JWTManager
from src.chemistry_lab.utils.exceptions import ValidationError, AuthenticationError


class TestUserAuthenticationConsistency:
    """用户认证一致性属性测试"""
    
    @pytest.fixture(scope="function")
    def auth_database(self):
        """创建认证测试数据库"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(bind=engine)
        session_factory = sessionmaker(bind=engine)
        
        yield session_factory, engine, db_path
        
        engine.dispose()
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture(scope="function")
    def user_manager(self, auth_database):
        """创建用户管理器实例"""
        session_factory, engine, db_path = auth_database
        
        user_manager = UserManager()
        user_manager.user_dao.db_manager._engine = engine
        user_manager.user_dao.db_manager._session_factory = session_factory
        
        return user_manager
    
    @given(
        username=st.text(min_size=3, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'), 
            whitelist_characters='_-'
        )),
        password=st.text(min_size=8, max_size=128),
        email=st.emails()
    )
    @settings(max_examples=50)
    def test_password_hash_consistency(self, username, password, email):
        """
        **特征: ai-chemistry-lab-assessment, 属性 1: 用户认证一致性**
        
        对于任何有效的密码，哈希后的密码应该能够通过验证，且每次哈希结果不同但都能验证成功。
        验证需求: 需求 1.4, 1.5
        """
        password_manager = PasswordManager()
        
        # 多次哈希同一密码
        hash1 = password_manager.hash_password(password)
        hash2 = password_manager.hash_password(password)
        hash3 = password_manager.hash_password(password)
        
        # 验证哈希结果不同（盐值不同）
        assert hash1 != hash2 != hash3, "相同密码的哈希结果应该不同"
        
        # 验证所有哈希都能通过验证
        assert password_manager.verify_password(password, hash1), "第一个哈希验证失败"
        assert password_manager.verify_password(password, hash2), "第二个哈希验证失败"
        assert password_manager.verify_password(password, hash3), "第三个哈希验证失败"
        
        # 验证错误密码不能通过验证
        wrong_password = password + "wrong"
        assert not password_manager.verify_password(wrong_password, hash1), "错误密码不应该通过验证"
        assert not password_manager.verify_password(wrong_password, hash2), "错误密码不应该通过验证"
        assert not password_manager.verify_password(wrong_password, hash3), "错误密码不应该通过验证"
    
    @given(
        user_data=st.fixed_dictionaries({
            'username': st.text(min_size=3, max_size=50, alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'), 
                whitelist_characters='_-'
            )),
            'password': st.text(min_size=8, max_size=128),
            'email': st.emails(),
            'is_admin': st.booleans(),
        })
    )
    @settings(max_examples=30)
    def test_jwt_token_consistency(self, user_data):
        """
        **特征: ai-chemistry-lab-assessment, 属性 1: 用户认证一致性**
        
        对于任何有效的用户档案，生成的JWT令牌应该能够正确验证并提取用户信息。
        验证需求: 需求 1.4, 1.5
        """
        jwt_manager = JWTManager()
        
        # 创建用户档案
        user_profile = UserProfile(
            user_id=12345,
            username=user_data['username'],
            email=user_data['email'],
            is_admin=user_data['is_admin']
        )
        
        # 生成令牌
        access_token = jwt_manager.generate_access_token(user_profile)
        refresh_token = jwt_manager.generate_refresh_token(user_profile)
        
        # 验证访问令牌
        access_payload = jwt_manager.verify_token(access_token)
        assert access_payload is not None, "访问令牌验证失败"
        assert access_payload['user_id'] == user_profile.user_id, "用户ID不匹配"
        assert access_payload['username'] == user_profile.username, "用户名不匹配"
        assert access_payload['is_admin'] == user_profile.is_admin, "管理员状态不匹配"
        assert access_payload['type'] == 'access', "令牌类型不正确"
        
        # 验证刷新令牌
        refresh_payload = jwt_manager.verify_token(refresh_token)
        assert refresh_payload is not None, "刷新令牌验证失败"
        assert refresh_payload['user_id'] == user_profile.user_id, "刷新令牌用户ID不匹配"
        assert refresh_payload['username'] == user_profile.username, "刷新令牌用户名不匹配"
        assert refresh_payload['type'] == 'refresh', "刷新令牌类型不正确"
        
        # 使用刷新令牌生成新的访问令牌
        new_access_token = jwt_manager.refresh_access_token(refresh_token)
        assert new_access_token is not None, "刷新访问令牌失败"
        
        # 验证新的访问令牌
        new_access_payload = jwt_manager.verify_token(new_access_token)
        assert new_access_payload is not None, "新访问令牌验证失败"
        assert new_access_payload['user_id'] == user_profile.user_id, "新访问令牌用户ID不匹配"
        assert new_access_payload['username'] == user_profile.username, "新访问令牌用户名不匹配"
    
    @given(
        registration_data=st.fixed_dictionaries({
            'username': st.text(min_size=3, max_size=50, alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'), 
                whitelist_characters='_-'
            )),
            'password': st.text(min_size=8, max_size=128).filter(
                lambda x: any(c.isupper() for c in x) and 
                         any(c.islower() for c in x) and 
                         any(c.isdigit() for c in x) and
                         any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in x)
            ),
            'email': st.emails(),
        })
    )
    @settings(max_examples=20)
    def test_user_registration_authentication_round_trip(self, user_manager, registration_data):
        """
        **特征: ai-chemistry-lab-assessment, 属性 1: 用户认证一致性**
        
        对于任何有效的注册数据，注册后应该能够使用相同凭据成功登录。
        验证需求: 需求 1.4, 1.5
        """
        # 创建注册数据
        reg_data = RegistrationData(
            username=registration_data['username'],
            password=registration_data['password'],
            email=registration_data['email']
        )
        
        # 注册用户
        registration_result = user_manager.register_user(reg_data)
        assert registration_result.success, f"用户注册失败: {registration_result.error_message}"
        assert registration_result.user_profile is not None, "注册后用户档案为空"
        assert registration_result.access_token is not None, "注册后访问令牌为空"
        
        # 验证用户档案信息
        user_profile = registration_result.user_profile
        assert user_profile.username == registration_data['username'], "注册后用户名不匹配"
        assert user_profile.email == registration_data['email'], "注册后邮箱不匹配"
        assert user_profile.is_active, "注册后用户应该是激活状态"
        assert not user_profile.is_admin, "注册后用户不应该是管理员"
        
        # 使用相同凭据登录
        login_credentials = LoginCredentials(
            username=registration_data['username'],
            password=registration_data['password'],
            remember_me=True
        )
        
        authentication_result = user_manager.authenticate_user(login_credentials)
        assert authentication_result.success, f"用户登录失败: {authentication_result.error_message}"
        assert authentication_result.user_profile is not None, "登录后用户档案为空"
        assert authentication_result.access_token is not None, "登录后访问令牌为空"
        assert authentication_result.refresh_token is not None, "记住密码时应该有刷新令牌"
        
        # 验证登录后的用户档案与注册时一致
        login_profile = authentication_result.user_profile
        assert login_profile.user_id == user_profile.user_id, "登录后用户ID不匹配"
        assert login_profile.username == user_profile.username, "登录后用户名不匹配"
        assert login_profile.email == user_profile.email, "登录后邮箱不匹配"
        assert login_profile.is_active == user_profile.is_active, "登录后激活状态不匹配"
        assert login_profile.is_admin == user_profile.is_admin, "登录后管理员状态不匹配"
    
    @given(
        username=st.text(min_size=3, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'), 
            whitelist_characters='_-'
        )),
        correct_password=st.text(min_size=8, max_size=128).filter(
            lambda x: any(c.isupper() for c in x) and 
                     any(c.islower() for c in x) and 
                     any(c.isdigit() for c in x) and
                     any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in x)
        ),
        wrong_password=st.text(min_size=1, max_size=128)
    )
    @settings(max_examples=30)
    def test_authentication_failure_consistency(self, user_manager, username, correct_password, wrong_password):
        """
        **特征: ai-chemistry-lab-assessment, 属性 1: 用户认证一致性**
        
        对于任何错误的密码，认证应该始终失败，且不泄露用户是否存在的信息。
        验证需求: 需求 1.4
        """
        assume(correct_password != wrong_password)
        
        # 先注册一个用户
        reg_data = RegistrationData(
            username=username,
            password=correct_password,
            email=f"{username}@test.com"
        )
        
        registration_result = user_manager.register_user(reg_data)
        assume(registration_result.success)  # 假设注册成功
        
        # 使用错误密码尝试登录
        wrong_credentials = LoginCredentials(
            username=username,
            password=wrong_password
        )
        
        auth_result = user_manager.authenticate_user(wrong_credentials)
        
        # 验证认证失败
        assert not auth_result.success, "错误密码不应该认证成功"
        assert auth_result.user_profile is None, "认证失败时不应该返回用户档案"
        assert auth_result.access_token is None, "认证失败时不应该返回访问令牌"
        assert auth_result.refresh_token is None, "认证失败时不应该返回刷新令牌"
        assert auth_result.error_code == "INVALID_CREDENTIALS", "错误代码应该是INVALID_CREDENTIALS"
        
        # 使用不存在的用户名尝试登录
        nonexistent_credentials = LoginCredentials(
            username=username + "_nonexistent",
            password=correct_password
        )
        
        nonexistent_result = user_manager.authenticate_user(nonexistent_credentials)
        
        # 验证不存在用户的认证失败
        assert not nonexistent_result.success, "不存在的用户不应该认证成功"
        assert nonexistent_result.error_code == "INVALID_CREDENTIALS", "不存在用户的错误代码应该相同"
        
        # 验证错误消息一致性（不泄露用户是否存在）
        assert auth_result.error_message == nonexistent_result.error_message, (
            "错误密码和不存在用户的错误消息应该相同，避免泄露用户存在信息"
        )
    
    @given(
        user_data=st.fixed_dictionaries({
            'username': st.text(min_size=3, max_size=50, alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'), 
                whitelist_characters='_-'
            )),
            'password': st.text(min_size=8, max_size=128).filter(
                lambda x: any(c.isupper() for c in x) and 
                         any(c.islower() for c in x) and 
                         any(c.isdigit() for c in x) and
                         any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in x)
            ),
            'email': st.emails(),
        })
    )
    @settings(max_examples=20)
    def test_token_verification_consistency(self, user_manager, user_data):
        """
        **特征: ai-chemistry-lab-assessment, 属性 1: 用户认证一致性**
        
        对于任何有效的用户，生成的令牌应该能够正确验证并返回一致的用户信息。
        验证需求: 需求 1.4, 1.5
        """
        # 注册用户
        reg_data = RegistrationData(
            username=user_data['username'],
            password=user_data['password'],
            email=user_data['email']
        )
        
        registration_result = user_manager.register_user(reg_data)
        assume(registration_result.success)
        
        original_profile = registration_result.user_profile
        access_token = registration_result.access_token
        
        # 通过令牌验证获取用户档案
        verified_profile = user_manager.verify_access_token(access_token)
        
        assert verified_profile is not None, "令牌验证应该返回用户档案"
        assert verified_profile.user_id == original_profile.user_id, "验证后用户ID应该一致"
        assert verified_profile.username == original_profile.username, "验证后用户名应该一致"
        assert verified_profile.email == original_profile.email, "验证后邮箱应该一致"
        assert verified_profile.is_active == original_profile.is_active, "验证后激活状态应该一致"
        assert verified_profile.is_admin == original_profile.is_admin, "验证后管理员状态应该一致"
        
        # 验证无效令牌
        invalid_token = access_token + "invalid"
        invalid_profile = user_manager.verify_access_token(invalid_token)
        assert invalid_profile is None, "无效令牌不应该返回用户档案"
    
    @given(
        users_data=st.lists(
            st.fixed_dictionaries({
                'username': st.text(min_size=3, max_size=50, alphabet=st.characters(
                    whitelist_categories=('Lu', 'Ll', 'Nd'), 
                    whitelist_characters='_-'
                )),
                'password': st.text(min_size=8, max_size=128).filter(
                    lambda x: any(c.isupper() for c in x) and 
                             any(c.islower() for c in x) and 
                             any(c.isdigit() for c in x) and
                             any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in x)
                ),
            }),
            min_size=2,
            max_size=10,
            unique_by=lambda x: x['username']
        )
    )
    @settings(max_examples=10)
    def test_multiple_users_isolation(self, user_manager, users_data):
        """
        **特征: ai-chemistry-lab-assessment, 属性 1: 用户认证一致性**
        
        对于任何多个用户，每个用户的认证应该是独立的，不会相互影响。
        验证需求: 需求 1.4, 1.5
        """
        registered_users = []
        
        # 注册所有用户
        for i, user_data in enumerate(users_data):
            reg_data = RegistrationData(
                username=user_data['username'],
                password=user_data['password'],
                email=f"{user_data['username']}@test{i}.com"
            )
            
            result = user_manager.register_user(reg_data)
            if result.success:
                registered_users.append((user_data, result))
        
        assume(len(registered_users) >= 2)  # 至少需要2个成功注册的用户
        
        # 验证每个用户都能独立认证
        for user_data, registration_result in registered_users:
            credentials = LoginCredentials(
                username=user_data['username'],
                password=user_data['password']
            )
            
            auth_result = user_manager.authenticate_user(credentials)
            assert auth_result.success, f"用户 {user_data['username']} 认证失败"
            
            # 验证用户信息一致性
            assert auth_result.user_profile.user_id == registration_result.user_profile.user_id
            assert auth_result.user_profile.username == user_data['username']
        
        # 验证用户间的令牌隔离
        tokens = [result.access_token for _, result in registered_users]
        user_ids = [result.user_profile.user_id for _, result in registered_users]
        
        # 每个令牌只能验证对应的用户
        for i, token in enumerate(tokens):
            verified_profile = user_manager.verify_access_token(token)
            assert verified_profile is not None, f"令牌 {i} 验证失败"
            assert verified_profile.user_id == user_ids[i], f"令牌 {i} 返回了错误的用户ID"
            
            # 验证令牌不能用于其他用户
            for j, other_user_id in enumerate(user_ids):
                if i != j:
                    # 这里我们通过用户ID来验证隔离性
                    assert verified_profile.user_id != other_user_id, (
                        f"令牌 {i} 不应该验证为用户 {j}"
                    )
    
    def test_password_strength_consistency(self):
        """
        **特征: ai-chemistry-lab-assessment, 属性 1: 用户认证一致性**
        
        密码可用性检查应该对相同的密码返回一致的结果。
        验证需求: 需求 1.4
        """
        password_manager = PasswordManager()
        
        test_passwords = [
            ("weak", True),
            ("123", True),
            ("WeakPassword", True),
            ("Weak123!", True),
            ("", False),
        ]
        
        for password, expected_strong in test_passwords:
            # 多次检查同一密码
            for _ in range(5):
                is_strong, suggestions = password_manager.check_password_strength(password)
                assert is_strong == expected_strong, (
                    f"密码 '{password}' 的强度检查结果不一致: 期望 {expected_strong}, 实际 {is_strong}"
                )
                
                # 验证错误信息的一致性
                if not is_strong:
                    assert len(suggestions) > 0, f"空密码 '{password}' 应有错误提示"
                else:
                    assert len(suggestions) == 0, f"可用密码 '{password}' 不应有错误提示"
