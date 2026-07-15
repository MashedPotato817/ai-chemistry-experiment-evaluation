#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库操作属性测试

**特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
验证数据库操作的数据持久化一致性属性。
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 导入被测试的模块
from src.chemistry_lab.database.models import Base, User, ExperimentSession, ExperimentStep, InteractionLog, ExperimentStatistics
from src.chemistry_lab.database.database import DatabaseManager
from src.chemistry_lab.utils.exceptions import ValidationError, DatabaseError


class TestDatabasePersistenceConsistency:
    """数据库持久化一致性属性测试"""
    
    @pytest.fixture(scope="function")
    def temp_database(self):
        """创建临时数据库用于测试"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # 创建测试数据库引擎
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(bind=engine)
        
        # 创建会话工厂
        session_factory = sessionmaker(bind=engine)
        
        yield session_factory, engine
        
        # 清理
        engine.dispose()
        Path(db_path).unlink(missing_ok=True)
    
    @given(
        username=st.text(min_size=3, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'), 
            whitelist_characters='_-'
        )),
        email=st.emails(),
        is_active=st.booleans(),
        is_admin=st.booleans()
    )
    @settings(max_examples=50)
    def test_user_crud_consistency(self, temp_database, username, email, is_active, is_admin):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何有效的用户数据，创建、读取、更新、删除操作应该保持数据一致性。
        验证需求: 需求 5.1, 5.2, 5.4
        """
        session_factory, engine = temp_database
        
        with session_factory() as session:
            # 创建用户
            user = User(
                username=username,
                password_hash="hashed_password_123",
                email=email,
                is_active=is_active,
                is_admin=is_admin
            )
            
            session.add(user)
            session.commit()
            user_id = user.user_id
            
            # 验证创建后的数据
            assert user_id is not None
            assert user.username == username
            assert user.email == email
            assert user.is_active == is_active
            assert user.is_admin == is_admin
            
            # 读取用户
            retrieved_user = session.query(User).filter_by(user_id=user_id).first()
            assert retrieved_user is not None
            assert retrieved_user.username == username
            assert retrieved_user.email == email
            assert retrieved_user.is_active == is_active
            assert retrieved_user.is_admin == is_admin
            
            # 更新用户
            new_email = f"updated_{email}"
            retrieved_user.email = new_email
            session.commit()
            
            # 验证更新
            updated_user = session.query(User).filter_by(user_id=user_id).first()
            assert updated_user.email == new_email
            
            # 删除用户
            session.delete(updated_user)
            session.commit()
            
            # 验证删除
            deleted_user = session.query(User).filter_by(user_id=user_id).first()
            assert deleted_user is None
    
    @given(
        experiment_type=st.text(min_size=1, max_size=100),
        duration_minutes=st.floats(min_value=1.0, max_value=180.0, allow_nan=False),
        s1_score=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
        s2_score=st.floats(min_value=0.0, max_value=2.0, allow_nan=False)
    )
    @settings(max_examples=50)
    def test_experiment_session_consistency(self, temp_database, experiment_type, duration_minutes, s1_score, s2_score):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何有效的实验会话数据，存储和检索应该保持数据一致性。
        验证需求: 需求 5.1, 5.2, 5.4
        """
        session_factory, engine = temp_database
        
        with session_factory() as session:
            # 先创建用户
            user = User(
                username="test_user_" + str(hash(experiment_type))[-6:],
                password_hash="hashed_password_123"
            )
            session.add(user)
            session.commit()
            
            # 创建实验会话
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes)
            total_duration = duration_minutes * 60  # 转换为秒
            final_score = 0.7 * s1_score + 0.3 * s2_score
            
            experiment_session = ExperimentSession(
                user_id=user.user_id,
                experiment_type=experiment_type,
                start_time=start_time,
                end_time=end_time,
                total_duration=total_duration,
                s1_score=s1_score,
                s2_score=s2_score,
                final_score=final_score,
                status="completed"
            )
            
            session.add(experiment_session)
            session.commit()
            session_id = experiment_session.session_id
            
            # 验证存储的数据
            retrieved_session = session.query(ExperimentSession).filter_by(session_id=session_id).first()
            assert retrieved_session is not None
            assert retrieved_session.experiment_type == experiment_type
            assert retrieved_session.user_id == user.user_id
            assert abs(retrieved_session.total_duration - total_duration) < 0.001
            assert abs(retrieved_session.s1_score - s1_score) < 0.001
            assert abs(retrieved_session.s2_score - s2_score) < 0.001
            assert abs(retrieved_session.final_score - final_score) < 0.001
            assert retrieved_session.status == "completed"
            
            # 验证计算属性
            calculated_duration = retrieved_session.calculate_duration()
            assert abs(calculated_duration - total_duration) < 1.0  # 允许1秒误差
            assert retrieved_session.is_completed == True
    
    @given(
        steps_data=st.lists(
            st.fixed_dictionaries({
                'step_name': st.text(min_size=1, max_size=100),
                'start_timestamp': st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
                'duration': st.floats(min_value=1.0, max_value=300.0, allow_nan=False),
                'sequence_order': st.integers(min_value=1, max_value=20),
            }),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=30)
    def test_experiment_steps_temporal_consistency(self, temp_database, steps_data):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何实验步骤序列，时间戳应该保持逻辑一致性。
        验证需求: 需求 5.1, 5.2, 需求 3.4, 需求 8.3
        """
        session_factory, engine = temp_database
        
        # 确保步骤顺序唯一且时间戳合理
        unique_orders = set()
        filtered_steps = []
        current_time = 0.0
        
        for step_data in sorted(steps_data, key=lambda x: x['sequence_order']):
            if step_data['sequence_order'] not in unique_orders:
                unique_orders.add(step_data['sequence_order'])
                # 调整时间戳确保逻辑一致性
                step_data['start_timestamp'] = current_time
                step_data['end_timestamp'] = current_time + step_data['duration']
                current_time = step_data['end_timestamp']
                filtered_steps.append(step_data)
        
        assume(len(filtered_steps) > 0)
        
        with session_factory() as session:
            # 创建用户和实验会话
            user = User(username="test_user_steps", password_hash="hashed_password_123")
            session.add(user)
            session.commit()
            
            experiment_session = ExperimentSession(
                user_id=user.user_id,
                experiment_type="测试实验",
                start_time=datetime.now(),
                status="active"
            )
            session.add(experiment_session)
            session.commit()
            
            # 创建实验步骤
            created_steps = []
            for step_data in filtered_steps:
                step = ExperimentStep(
                    session_id=experiment_session.session_id,
                    step_name=step_data['step_name'],
                    start_timestamp=step_data['start_timestamp'],
                    end_timestamp=step_data['end_timestamp'],
                    duration=step_data['duration'],
                    sequence_order=step_data['sequence_order']
                )
                session.add(step)
                created_steps.append(step)
            
            session.commit()
            
            # 验证时间戳一致性
            retrieved_steps = session.query(ExperimentStep).filter_by(
                session_id=experiment_session.session_id
            ).order_by(ExperimentStep.sequence_order).all()
            
            assert len(retrieved_steps) == len(filtered_steps)
            
            for i, step in enumerate(retrieved_steps):
                original_data = filtered_steps[i]
                
                # 验证基本数据一致性
                assert step.step_name == original_data['step_name']
                assert abs(step.start_timestamp - original_data['start_timestamp']) < 0.001
                assert abs(step.end_timestamp - original_data['end_timestamp']) < 0.001
                assert abs(step.duration - original_data['duration']) < 0.001
                assert step.sequence_order == original_data['sequence_order']
                
                # 验证时间戳逻辑一致性
                assert step.start_timestamp < step.end_timestamp
                assert abs((step.end_timestamp - step.start_timestamp) - step.duration) < 0.001
                
                # 验证步骤间的时间顺序
                if i > 0:
                    prev_step = retrieved_steps[i - 1]
                    assert step.start_timestamp >= prev_step.end_timestamp
                    assert step.sequence_order > prev_step.sequence_order
    
    @given(
        interaction_data=st.lists(
            st.fixed_dictionaries({
                'interaction_type': st.sampled_from(['question', 'guidance', 'warning', 'feedback']),
                'timestamp': st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
                'user_input': st.text(max_size=500),
                'system_response': st.text(max_size=1000),
                'response_time': st.floats(min_value=10.0, max_value=5000.0, allow_nan=False),
            }),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=30)
    def test_interaction_logs_consistency(self, temp_database, interaction_data):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何交互日志数据，存储和检索应该保持完整性和时间顺序。
        验证需求: 需求 5.1, 5.2, 5.4
        """
        session_factory, engine = temp_database
        
        with session_factory() as session:
            # 创建用户和实验会话
            user = User(username="test_user_interactions", password_hash="hashed_password_123")
            session.add(user)
            session.commit()
            
            experiment_session = ExperimentSession(
                user_id=user.user_id,
                experiment_type="交互测试实验",
                start_time=datetime.now(),
                status="active"
            )
            session.add(experiment_session)
            session.commit()
            
            # 创建交互日志
            for interaction in interaction_data:
                log = InteractionLog(
                    session_id=experiment_session.session_id,
                    timestamp=interaction['timestamp'],
                    interaction_type=interaction['interaction_type'],
                    user_input=interaction['user_input'],
                    system_response=interaction['system_response'],
                    response_time=interaction['response_time']
                )
                session.add(log)
            
            session.commit()
            
            # 验证数据一致性
            retrieved_logs = session.query(InteractionLog).filter_by(
                session_id=experiment_session.session_id
            ).order_by(InteractionLog.timestamp).all()
            
            assert len(retrieved_logs) == len(interaction_data)
            
            # 按时间戳排序原始数据进行比较
            sorted_original = sorted(interaction_data, key=lambda x: x['timestamp'])
            
            for i, log in enumerate(retrieved_logs):
                original = sorted_original[i]
                
                assert log.interaction_type == original['interaction_type']
                assert abs(log.timestamp - original['timestamp']) < 0.001
                assert log.user_input == original['user_input']
                assert log.system_response == original['system_response']
                assert abs(log.response_time - original['response_time']) < 0.001
                
                # 验证时间戳顺序
                if i > 0:
                    prev_log = retrieved_logs[i - 1]
                    assert log.timestamp >= prev_log.timestamp
    
    @given(
        experiment_type=st.text(min_size=1, max_size=100),
        mean_duration=st.floats(min_value=60.0, max_value=7200.0, allow_nan=False),
        std_deviation=st.floats(min_value=1.0, max_value=1800.0, allow_nan=False),
        sample_count=st.integers(min_value=1, max_value=1000),
        mean_score=st.floats(min_value=0.0, max_value=2.0, allow_nan=False)
    )
    @settings(max_examples=30)
    def test_experiment_statistics_consistency(self, temp_database, experiment_type, mean_duration, std_deviation, sample_count, mean_score):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何实验统计数据，存储和更新应该保持数据一致性。
        验证需求: 需求 5.1, 5.2, 5.4, 需求 7.4
        """
        session_factory, engine = temp_database
        
        with session_factory() as session:
            # 创建统计记录
            stats = ExperimentStatistics(
                experiment_type=experiment_type,
                mean_duration=mean_duration,
                std_deviation=std_deviation,
                sample_count=sample_count,
                mean_score=mean_score
            )
            
            session.add(stats)
            session.commit()
            stat_id = stats.stat_id
            
            # 验证创建的数据
            retrieved_stats = session.query(ExperimentStatistics).filter_by(stat_id=stat_id).first()
            assert retrieved_stats is not None
            assert retrieved_stats.experiment_type == experiment_type
            assert abs(retrieved_stats.mean_duration - mean_duration) < 0.001
            assert abs(retrieved_stats.std_deviation - std_deviation) < 0.001
            assert retrieved_stats.sample_count == sample_count
            assert abs(retrieved_stats.mean_score - mean_score) < 0.001
            
            # 更新统计数据
            new_sample_count = sample_count + 10
            new_mean_duration = mean_duration * 1.1
            
            retrieved_stats.sample_count = new_sample_count
            retrieved_stats.mean_duration = new_mean_duration
            session.commit()
            
            # 验证更新后的数据
            updated_stats = session.query(ExperimentStatistics).filter_by(stat_id=stat_id).first()
            assert updated_stats.sample_count == new_sample_count
            assert abs(updated_stats.mean_duration - new_mean_duration) < 0.001
            
            # 验证last_updated字段被自动更新
            assert updated_stats.last_updated > updated_stats.created_at
    
    def test_database_transaction_consistency(self, temp_database):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        验证数据库事务的一致性，确保失败时能正确回滚。
        验证需求: 需求 5.2, 5.5
        """
        session_factory, engine = temp_database
        
        with session_factory() as session:
            # 创建用户
            user = User(username="transaction_test_user", password_hash="hashed_password_123")
            session.add(user)
            session.commit()
            
            # 测试事务回滚
            try:
                # 开始事务
                experiment1 = ExperimentSession(
                    user_id=user.user_id,
                    experiment_type="事务测试1",
                    start_time=datetime.now(),
                    status="active"
                )
                session.add(experiment1)
                
                # 故意创建一个会导致错误的记录（违反唯一约束）
                user_duplicate = User(username="transaction_test_user", password_hash="another_hash")
                session.add(user_duplicate)
                
                # 这里应该失败并回滚
                session.commit()
                
            except Exception:
                # 验证回滚后的状态
                session.rollback()
                
                # 检查实验会话是否被回滚
                experiments = session.query(ExperimentSession).filter_by(user_id=user.user_id).all()
                assert len(experiments) == 0, "事务回滚失败，实验会话仍然存在"
                
                # 检查用户是否仍然存在（之前已提交）
                existing_user = session.query(User).filter_by(username="transaction_test_user").first()
                assert existing_user is not None, "原有用户不应该被回滚"
    
    def test_foreign_key_consistency(self, temp_database):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        验证外键约束的一致性，确保数据完整性。
        验证需求: 需求 5.1, 5.2, 5.4
        """
        session_factory, engine = temp_database
        
        with session_factory() as session:
            # 创建用户
            user = User(username="fk_test_user", password_hash="hashed_password_123")
            session.add(user)
            session.commit()
            
            # 创建实验会话
            experiment_session = ExperimentSession(
                user_id=user.user_id,
                experiment_type="外键测试实验",
                start_time=datetime.now(),
                status="active"
            )
            session.add(experiment_session)
            session.commit()
            
            # 创建实验步骤
            step = ExperimentStep(
                session_id=experiment_session.session_id,
                step_name="测试步骤",
                start_timestamp=0.0,
                end_timestamp=60.0,
                duration=60.0,
                sequence_order=1
            )
            session.add(step)
            session.commit()
            
            # 验证关联关系
            retrieved_session = session.query(ExperimentSession).filter_by(
                session_id=experiment_session.session_id
            ).first()
            assert retrieved_session.user.username == "fk_test_user"
            assert len(retrieved_session.experiment_steps) == 1
            assert retrieved_session.experiment_steps[0].step_name == "测试步骤"
            
            # 测试级联删除
            session.delete(retrieved_session)
            session.commit()
            
            # 验证级联删除
            remaining_steps = session.query(ExperimentStep).filter_by(
                session_id=experiment_session.session_id
            ).all()
            assert len(remaining_steps) == 0, "级联删除失败，实验步骤仍然存在"