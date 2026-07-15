#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
装饰器工具

提供常用的装饰器函数，用于重试、计时、验证等功能。
"""

import time
import functools
from typing import Any, Callable, Optional, Type, Union, List
from .logger import get_logger
from .exceptions import ChemistryLabException

logger = get_logger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Union[Type[Exception], tuple] = Exception,
    on_failure: Optional[Callable] = None,
):
    """
    重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间(秒)
        backoff: 延迟时间倍数
        exceptions: 需要重试的异常类型
        on_failure: 失败时的回调函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        # 最后一次尝试失败
                        logger.error(
                            f"函数 {func.__name__} 重试 {max_attempts} 次后仍然失败: {e}"
                        )
                        if on_failure:
                            on_failure(e)
                        raise
                    
                    logger.warning(
                        f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}, "
                        f"{current_delay:.1f}秒后重试"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # 理论上不会到达这里
            raise last_exception
        
        return wrapper
    return decorator


def timing(log_level: str = "INFO"):
    """
    计时装饰器
    
    Args:
        log_level: 日志级别
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                log_func = getattr(logger, log_level.lower())
                log_func(f"函数 {func.__name__} 执行时间: {execution_time:.3f}秒")
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"函数 {func.__name__} 执行失败 (耗时: {execution_time:.3f}秒): {e}"
                )
                raise
        
        return wrapper
    return decorator


def validate_input(**validators):
    """
    输入验证装饰器
    
    Args:
        **validators: 参数名到验证函数的映射
    
    Example:
        @validate_input(
            username=lambda x: len(x) >= 3,
            age=lambda x: x >= 0
        )
        def create_user(username: str, age: int):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数参数名
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 验证参数
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        if not validator(value):
                            raise ValueError(f"参数 {param_name} 验证失败: {value}")
                    except Exception as e:
                        logger.error(f"参数验证异常: {param_name}={value}, 错误: {e}")
                        raise ChemistryLabException(
                            f"参数验证失败: {param_name}",
                            error_code="VALIDATION_ERROR",
                            details={"parameter": param_name, "value": str(value)}
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def cache_result(ttl: Optional[float] = None):
    """
    结果缓存装饰器
    
    Args:
        ttl: 缓存生存时间(秒)，None表示永久缓存
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # 检查缓存
            if cache_key in cache:
                cached_result, cached_time = cache[cache_key]
                if ttl is None or (current_time - cached_time) < ttl:
                    logger.debug(f"函数 {func.__name__} 使用缓存结果")
                    return cached_result
                else:
                    # 缓存过期，删除
                    del cache[cache_key]
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache[cache_key] = (result, current_time)
            logger.debug(f"函数 {func.__name__} 结果已缓存")
            
            return result
        
        # 添加清除缓存的方法
        wrapper.clear_cache = lambda: cache.clear()
        wrapper.cache_info = lambda: {
            "cache_size": len(cache),
            "cached_keys": list(cache.keys())
        }
        
        return wrapper
    return decorator


def async_to_sync(func: Callable) -> Callable:
    """
    将异步函数转换为同步函数
    """
    import asyncio
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(func(*args, **kwargs))
    
    return wrapper