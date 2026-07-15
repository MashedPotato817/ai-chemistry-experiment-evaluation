#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发数据访问属性测试

**特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
验证并发环境下数据库操作的一致性属性。
"""

import pytest
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from hypothesis import given, strategies as st, settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 导入被测试的模块
from src.chemistry_lab.database.models import Base, User, ExperimentSession, ExperimentStep
from src.chemistry_lab.database.dao import UserDAO, ExperimentDAO, StatisticsDAO
from src.chemistry_lab.database.database import DatabaseManager
from src.chemistry_lab.utils.exceptions import DatabaseError, ValidationError


class TestConcurrentDatabaseAccess:
    """并发数据库访问一致性测试"""
    
    @pytest.fixture(scope="function")
    def concurrent_database(self):
        """创建支持并发访问的临时数据库"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # 创建测试数据库引擎，启用WAL模式支持并发
        engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            pool_pre_ping=True,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            }
        )
        
        # 设置WAL模式
        with engine.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
        
        Base.metadata.create_all(bind=engine)
        session_factory = sessionmaker(bind=engine)
        
        yield session_factory, engine, db_path
        
        # 清理
        engine.dispose()
        Path(db_path).unlink(missing_ok=True)
    
    @given(
        num_threads=st.integers(min_value=2, max_value=10),
        users_per_thread=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=10, deadline=30000)  # 增加超时时间
    def test_concurrent_user_creation_consistency(self, concurrent_database, num_threads, users_per_thread):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何并发用户创建操作，应该保持数据一致性，不出现重复用户名。
        验证需求: 需求 5.2
        """
        session_factory, engine, db_path = concurrent_database
        
        # 创建用户DAO实例
        user_dao = UserDAO()
        user_dao.db_manager._engine = engine
        user_dao.db_manager._session_factory = session_factory
        
        created_users = []
        errors = []
        lock = threading.Lock()
        
        def create_users_batch(thread_id: int):
            """在单个线程中创建用户批次"""
            thread_users = []
            thread_errors = []
            
            for i in range(users_per_thread):
                try:
                    username = f"user_{thread_id}_{i}"
                    password_hash = f"hash_{thread_id}_{i}"
                    email = f"user_{thread_id}_{i}@test.com"
                    
                    user = user_dao.create_user(username, password_hash, email)
                    thread_users.append(user)
                    
                except Exception as e:
                    thread_errors.append((username, str(e)))
            
            # 线程安全地添加结果
            with lock:
                created_users.extend(thread_users)
                errors.extend(thread_errors)
        
        # 并发执行用户创建
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(create_users_batch, thread_id)
                for thread_id in range(num_threads)
            ]
            
            # 等待所有线程完成
            for future in as_completed(futures):
                future.result()  # 获取结果，如果有异常会抛出
        
        # 验证结果一致性
        expected_total = num_threads * users_per_thread
        
        # 所有用户都应该成功创建（没有重复用户名冲突）
        assert len(created_users) == expected_total, (
            f"创建的用户数量不正确: 期望 {expected_total}, 实际 {len(created_users)}, "
            f"错误: {errors}"
        )
        
        # 验证用户名唯一性
        usernames = [user.username for user in created_users]
        assert len(set(usernames)) == len(usernames), "存在重复的用户名"
        
        # 验证邮箱唯一性
        emails = [user.email for user in created_users if user.email]
        assert len(set(emails)) == len(emails), "存在重复的邮箱"
        
        # 验证数据库中的实际记录数
        with session_factory() as session:
            db_user_count = session.query(User).count()
            assert db_user_count == expected_total, (
                f"数据库中的用户数量不正确: 期望 {expected_total}, 实际 {db_user_count}"
            )
    
    @given(
        num_threads=st.integers(min_value=2, max_value=8),
        experiments_per_thread=st.integers(min_value=3, max_value=10)
    )
    @settings(max_examples=8, deadline=30000)
    def test_concurrent_experiment_creation_consistency(self, concurrent_database, num_threads, experiments_per_thread):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何并发实验会话创建操作，应该保持数据一致性和外键完整性。
        验证需求: 需求 5.2
        """
        session_factory, engine, db_path = concurrent_database
        
        # 创建DAO实例
        user_dao = UserDAO()
        experiment_dao = ExperimentDAO()
        
        # 设置数据库连接
        for dao in [user_dao, experiment_dao]:
            dao.db_manager._engine = engine
            dao.db_manager._session_factory = session_factory
        
        # 先创建测试用户
        test_users = []
        for i in range(num_threads):
            user = user_dao.create_user(f"test_user_{i}", f"hash_{i}", f"user_{i}@test.com")
            test_users.append(user)
        
        created_experiments = []
        errors = []
        lock = threading.Lock()
        
        def create_experiments_batch(thread_id: int):
            """在单个线程中创建实验会话批次"""
            thread_experiments = []
            thread_errors = []
            user = test_users[thread_id]
            
            for i in range(experiments_per_thread):
                try:
                    experiment_type = f"实验类型_{thread_id}_{i}"
                    
                    experiment = experiment_dao.create_experiment_session(
                        user_id=user.user_id,
                        experiment_type=experiment_type,
                        start_time=datetime.now()
                    )
                    thread_experiments.append(experiment)
                    
                except Exception as e:
                    thread_errors.append((experiment_type, str(e)))
            
            # 线程安全地添加结果
            with lock:
                created_experiments.extend(thread_experiments)
                errors.extend(thread_errors)
        
        # 并发执行实验创建
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(create_experiments_batch, thread_id)
                for thread_id in range(num_threads)
            ]
            
            for future in as_completed(futures):
                future.result()
        
        # 验证结果一致性
        expected_total = num_threads * experiments_per_thread
        
        assert len(created_experiments) == expected_total, (
            f"创建的实验数量不正确: 期望 {expected_total}, 实际 {len(created_experiments)}, "
            f"错误: {errors}"
        )
        
        # 验证外键关系完整性
        for experiment in created_experiments:
            assert experiment.user_id in [user.user_id for user in test_users]
        
        # 验证数据库中的实际记录数
        with session_factory() as session:
            db_experiment_count = session.query(ExperimentSession).count()
            assert db_experiment_count == expected_total
    
    @given(
        num_readers=st.integers(min_value=2, max_value=6),
        num_writers=st.integers(min_value=1, max_value=3),
        operations_per_thread=st.integers(min_value=5, max_value=15)
    )
    @settings(max_examples=5, deadline=45000)
    def test_concurrent_read_write_consistency(self, concurrent_database, num_readers, num_writers, operations_per_thread):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何并发读写操作，读取的数据应该保持一致性。
        验证需求: 需求 5.2
        """
        session_factory, engine, db_path = concurrent_database
        
        user_dao = UserDAO()
        user_dao.db_manager._engine = engine
        user_dao.db_manager._session_factory = session_factory
        
        # 创建初始用户
        initial_user = user_dao.create_user("initial_user", "initial_hash", "initial@test.com")
        
        read_results = []
        write_results = []
        errors = []
        lock = threading.Lock()
        
        def reader_thread(reader_id: int):
            """读取线程"""
            thread_reads = []
            thread_errors = []
            
            for i in range(operations_per_thread):
                try:
                    # 随机读取用户
                    user = user_dao.get_user_by_id(initial_user.user_id)
                    if user:
                        thread_reads.append({
                            'reader_id': reader_id,
                            'operation': i,
                            'username': user.username,
                            'email': user.email,
                            'is_active': user.is_active
                        })
                    time.sleep(0.01)  # 小延迟模拟真实场景
                    
                except Exception as e:
                    thread_errors.append(f"Reader {reader_id}: {str(e)}")
            
            with lock:
                read_results.extend(thread_reads)
                errors.extend(thread_errors)
        
        def writer_thread(writer_id: int):
            """写入线程"""
            thread_writes = []
            thread_errors = []
            
            for i in range(operations_per_thread):
                try:
                    # 更新用户信息
                    new_email = f"updated_{writer_id}_{i}@test.com"
                    updated_user = user_dao.update_user(
                        initial_user.user_id,
                        email=new_email
                    )
                    
                    if updated_user:
                        thread_writes.append({
                            'writer_id': writer_id,
                            'operation': i,
                            'new_email': new_email
                        })
                    
                    time.sleep(0.02)  # 小延迟
                    
                except Exception as e:
                    thread_errors.append(f"Writer {writer_id}: {str(e)}")
            
            with lock:
                write_results.extend(thread_writes)
                errors.extend(thread_errors)
        
        # 并发执行读写操作
        with ThreadPoolExecutor(max_workers=num_readers + num_writers) as executor:
            # 启动读取线程
            reader_futures = [
                executor.submit(reader_thread, i)
                for i in range(num_readers)
            ]
            
            # 启动写入线程
            writer_futures = [
                executor.submit(writer_thread, i)
                for i in range(num_writers)
            ]
            
            # 等待所有线程完成
            all_futures = reader_futures + writer_futures
            for future in as_completed(all_futures):
                future.result()
        
        # 验证一致性
        assert len(errors) == 0, f"并发操作出现错误: {errors}"
        
        # 验证读取操作的数据一致性
        # 所有读取到的用户名应该是一致的
        usernames = [result['username'] for result in read_results]
        assert all(username == 'initial_user' for username in usernames), "用户名读取不一致"
        
        # 验证最终状态
        final_user = user_dao.get_user_by_id(initial_user.user_id)
        assert final_user is not None, "最终用户不存在"
        assert final_user.username == 'initial_user', "最终用户名不正确"
        
        # 验证写入操作确实发生了
        assert len(write_results) == num_writers * operations_per_thread, "写入操作数量不正确"
    
    def test_concurrent_transaction_isolation(self, concurrent_database):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        验证并发事务的隔离性，确保事务回滚不影响其他事务。
        验证需求: 需求 5.2, 5.5
        """
        session_factory, engine, db_path = concurrent_database
        
        user_dao = UserDAO()
        user_dao.db_manager._engine = engine
        user_dao.db_manager._session_factory = session_factory
        
        success_count = 0
        failure_count = 0
        lock = threading.Lock()
        
        def transaction_thread(thread_id: int, should_fail: bool):
            """事务线程"""
            nonlocal success_count, failure_count
            
            try:
                if should_fail:
                    # 故意创建重复用户名导致失败
                    user_dao.create_user("duplicate_user", f"hash_{thread_id}")
                else:
                    # 正常创建用户
                    user_dao.create_user(f"normal_user_{thread_id}", f"hash_{thread_id}")
                
                with lock:
                    success_count += 1
                    
            except (DatabaseError, ValidationError):
                with lock:
                    failure_count += 1
        
        # 先创建一个用户，用于后续的重复用户名测试
        user_dao.create_user("duplicate_user", "original_hash")
        
        # 并发执行事务
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            
            # 3个正常事务
            for i in range(3):
                futures.append(executor.submit(transaction_thread, i, False))
            
            # 3个会失败的事务
            for i in range(3, 6):
                futures.append(executor.submit(transaction_thread, i, True))
            
            # 等待所有事务完成
            for future in as_completed(futures):
                future.result()
        
        # 验证事务隔离性
        assert success_count == 3, f"成功事务数量不正确: {success_count}"
        assert failure_count == 3, f"失败事务数量不正确: {failure_count}"
        
        # 验证数据库最终状态
        with session_factory() as session:
            total_users = session.query(User).count()
            # 应该有4个用户：1个原始的duplicate_user + 3个成功创建的normal_user
            assert total_users == 4, f"最终用户数量不正确: {total_users}"
            
            # 验证具体用户存在
            duplicate_user = session.query(User).filter_by(username="duplicate_user").first()
            assert duplicate_user is not None, "原始重复用户不存在"
            
            normal_users = session.query(User).filter(User.username.like("normal_user_%")).all()
            assert len(normal_users) == 3, f"正常用户数量不正确: {len(normal_users)}"
    
    @pytest.mark.slow
    def test_high_concurrency_stress(self, concurrent_database):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        高并发压力测试，验证系统在高负载下的数据一致性。
        验证需求: 需求 5.2
        """
        session_factory, engine, db_path = concurrent_database
        
        user_dao = UserDAO()
        experiment_dao = ExperimentDAO()
        
        for dao in [user_dao, experiment_dao]:
            dao.db_manager._engine = engine
            dao.db_manager._session_factory = session_factory
        
        # 高并发参数
        num_threads = 20
        operations_per_thread = 50
        
        results = []
        errors = []
        lock = threading.Lock()
        
        def stress_thread(thread_id: int):
            """压力测试线程"""
            thread_results = []
            thread_errors = []
            
            try:
                # 创建用户
                user = user_dao.create_user(
                    f"stress_user_{thread_id}",
                    f"stress_hash_{thread_id}",
                    f"stress_{thread_id}@test.com"
                )
                thread_results.append(('user_created', user.user_id))
                
                # 创建多个实验会话
                for i in range(operations_per_thread):
                    experiment = experiment_dao.create_experiment_session(
                        user_id=user.user_id,
                        experiment_type=f"压力测试实验_{i}",
                        start_time=datetime.now()
                    )
                    thread_results.append(('experiment_created', experiment.session_id))
                
            except Exception as e:
                thread_errors.append(f"Thread {thread_id}: {str(e)}")
            
            with lock:
                results.extend(thread_results)
                errors.extend(thread_errors)
        
        # 执行压力测试
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(stress_thread, i)
                for i in range(num_threads)
            ]
            
            for future in as_completed(futures):
                future.result()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证结果
        assert len(errors) == 0, f"压力测试出现错误: {errors[:10]}..."  # 只显示前10个错误
        
        # 验证操作数量
        user_operations = [r for r in results if r[0] == 'user_created']
        experiment_operations = [r for r in results if r[0] == 'experiment_created']
        
        assert len(user_operations) == num_threads, f"用户创建数量不正确: {len(user_operations)}"
        assert len(experiment_operations) == num_threads * operations_per_thread, (
            f"实验创建数量不正确: {len(experiment_operations)}"
        )
        
        # 验证数据库最终状态
        with session_factory() as session:
            db_users = session.query(User).count()
            db_experiments = session.query(ExperimentSession).count()
            
            assert db_users == num_threads, f"数据库用户数量不正确: {db_users}"
            assert db_experiments == num_threads * operations_per_thread, (
                f"数据库实验数量不正确: {db_experiments}"
            )
        
        # 性能验证（可选）
        operations_per_second = len(results) / execution_time
        print(f"压力测试完成: {len(results)} 操作在 {execution_time:.2f} 秒内完成")
        print(f"平均操作速度: {operations_per_second:.2f} 操作/秒")