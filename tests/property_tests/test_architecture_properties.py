#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目架构属性测试

**特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
验证系统架构的数据持久化一致性属性。
"""

import pytest
import tempfile
import json
from pathlib import Path
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, patch

# 导入被测试的模块
from src.chemistry_lab.config import Config
from src.chemistry_lab.utils.helpers import (
    ensure_dir, load_json, save_json, generate_id, 
    format_timestamp, safe_divide, clamp
)
from src.chemistry_lab.utils.exceptions import ChemistryLabException


class TestDataPersistenceConsistency:
    """数据持久化一致性属性测试"""
    
    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100)
    def test_json_round_trip_consistency(self, data):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何有效的数据字典，保存到JSON文件后再加载应该得到相同的数据。
        验证需求: 需求 5.1, 5.2
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # 保存数据
            save_json(data, temp_path)
            
            # 加载数据
            loaded_data = load_json(temp_path)
            
            # 验证数据一致性
            assert loaded_data == data, f"数据不一致: 原始={data}, 加载={loaded_data}"
            
        finally:
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()
    
    @given(
        directory_parts=st.lists(
            st.text(min_size=1, max_size=20, alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'), 
                whitelist_characters='_-'
            )),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=50)
    def test_directory_creation_consistency(self, directory_parts):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何有效的目录路径，ensure_dir函数应该能够创建目录并返回正确的Path对象。
        验证需求: 需求 5.1
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            target_path = base_path
            
            # 构建目录路径
            for part in directory_parts:
                target_path = target_path / part
            
            # 创建目录
            result_path = ensure_dir(target_path)
            
            # 验证目录存在且路径正确
            assert result_path.exists(), f"目录未创建: {result_path}"
            assert result_path.is_dir(), f"路径不是目录: {result_path}"
            assert result_path == target_path.resolve(), f"路径不匹配: {result_path} != {target_path}"
    
    @given(
        config_data=st.fixed_dictionaries({
            'app_name': st.text(min_size=1, max_size=100),
            'debug': st.booleans(),
            'log_level': st.sampled_from(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
        })
    )
    @settings(max_examples=50)
    def test_config_consistency(self, config_data):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何有效的配置数据，Config对象应该正确存储和返回配置值。
        验证需求: 需求 5.1, 5.2
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 创建配置对象
            config = Config(
                project_root=temp_path,
                app_name=config_data['app_name'],
                debug=config_data['debug'],
                log_level=config_data['log_level'],
                data_dir=temp_path / "data",
                models_dir=temp_path / "models",
                logs_dir=temp_path / "logs",
            )
            
            # 验证配置值一致性
            assert config.app_name == config_data['app_name']
            assert config.debug == config_data['debug']
            assert config.log_level == config_data['log_level']
            
            # 验证目录创建
            config.create_directories()
            assert config.data_dir.exists()
            assert config.models_dir.exists()
            assert config.logs_dir.exists()
    
    @given(
        prefix=st.text(min_size=0, max_size=20, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'), 
            whitelist_characters='_-'
        ))
    )
    @settings(max_examples=100)
    def test_id_generation_uniqueness(self, prefix):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何前缀，generate_id函数应该生成唯一的ID。
        验证需求: 需求 5.1
        """
        # 生成多个ID
        ids = set()
        for _ in range(100):
            new_id = generate_id(prefix)
            
            # 验证ID格式
            if prefix:
                assert new_id.startswith(f"{prefix}_"), f"ID格式错误: {new_id}"
            
            # 验证唯一性
            assert new_id not in ids, f"ID重复: {new_id}"
            ids.add(new_id)
    
    @given(
        timestamp=st.one_of(
            st.none(),
            st.floats(min_value=0, max_value=2147483647, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(max_examples=50)
    def test_timestamp_formatting_consistency(self, timestamp):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何有效的时间戳，format_timestamp函数应该返回格式正确的时间字符串。
        验证需求: 需求 5.1
        """
        result = format_timestamp(timestamp)
        
        # 验证返回值是字符串
        assert isinstance(result, str), f"返回值不是字符串: {type(result)}"
        
        # 验证格式 (YYYY-MM-DD HH:MM:SS)
        import re
        pattern = r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'
        assert re.match(pattern, result), f"时间格式错误: {result}"
    
    @given(
        numerator=st.floats(allow_nan=False, allow_infinity=False),
        denominator=st.floats(allow_nan=False, allow_infinity=False),
        default=st.floats(allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_safe_divide_consistency(self, numerator, denominator, default):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何数值输入，safe_divide函数应该返回有效的数值结果。
        验证需求: 需求 5.1
        """
        result = safe_divide(numerator, denominator, default)
        
        # 验证返回值是数值
        assert isinstance(result, (int, float)), f"返回值不是数值: {type(result)}"
        
        # 验证逻辑正确性
        if denominator == 0:
            assert result == default, f"除零时应返回默认值: {result} != {default}"
        else:
            expected = numerator / denominator
            assert abs(result - expected) < 1e-10, f"计算错误: {result} != {expected}"
    
    @given(
        value=st.floats(allow_nan=False, allow_infinity=False),
        min_value=st.floats(allow_nan=False, allow_infinity=False),
        max_value=st.floats(allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_clamp_function_consistency(self, value, min_value, max_value):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        对于任何数值输入，clamp函数应该正确限制值的范围。
        验证需求: 需求 5.1
        """
        # 确保min_value <= max_value
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        
        result = clamp(value, min_value, max_value)
        
        # 验证结果在范围内
        assert min_value <= result <= max_value, (
            f"结果超出范围: {result} not in [{min_value}, {max_value}]"
        )
        
        # 验证逻辑正确性
        if value < min_value:
            assert result == min_value, f"应该返回最小值: {result} != {min_value}"
        elif value > max_value:
            assert result == max_value, f"应该返回最大值: {result} != {max_value}"
        else:
            assert result == value, f"应该返回原值: {result} != {value}"


class TestSystemIntegrity:
    """系统完整性测试"""
    
    def test_module_imports_consistency(self):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        验证所有核心模块都能正确导入，确保系统架构完整性。
        验证需求: 需求 5.1
        """
        # 测试配置模块
        from src.chemistry_lab.config import Config, config
        assert isinstance(config, Config)
        
        # 测试工具模块
        from src.chemistry_lab.utils import (
            setup_logger, get_logger, ChemistryLabException,
            ensure_dir, load_json, save_json, generate_id
        )
        
        # 验证函数可调用
        assert callable(setup_logger)
        assert callable(get_logger)
        assert callable(ensure_dir)
        assert callable(load_json)
        assert callable(save_json)
        assert callable(generate_id)
        
        # 验证异常类
        assert issubclass(ChemistryLabException, Exception)
    
    def test_directory_structure_consistency(self, test_config):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        验证项目目录结构的一致性和完整性。
        验证需求: 需求 5.1
        """
        # 验证配置目录存在
        assert test_config.data_dir.exists()
        assert test_config.models_dir.exists()
        assert test_config.logs_dir.exists()
        
        # 验证目录是可写的
        test_file = test_config.data_dir / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()
        assert test_file.read_text() == "test"
        test_file.unlink()
    
    @pytest.mark.parametrize("error_class,error_code", [
        ("DatabaseError", "DATABASE_ERROR"),
        ("ModelError", "MODEL_ERROR"),
        ("APIError", "API_ERROR"),
        ("ValidationError", "VALIDATION_ERROR"),
    ])
    def test_exception_consistency(self, error_class, error_code):
        """
        **特征: ai-chemistry-lab-assessment, 属性 5: 数据持久化一致性**
        
        验证自定义异常类的一致性和正确性。
        验证需求: 需求 5.1, 5.5
        """
        from src.chemistry_lab.utils.exceptions import (
            DatabaseError, ModelError, APIError, ValidationError
        )
        
        # 获取异常类
        exception_classes = {
            "DatabaseError": DatabaseError,
            "ModelError": ModelError,
            "APIError": APIError,
            "ValidationError": ValidationError,
        }
        
        ExceptionClass = exception_classes[error_class]
        
        # 创建异常实例
        exception = ExceptionClass("测试错误消息")
        
        # 验证异常属性
        assert exception.error_code == error_code
        assert exception.message == "测试错误消息"
        assert isinstance(exception.details, dict)
        
        # 验证异常可以被抛出和捕获
        with pytest.raises(ExceptionClass):
            raise exception