#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义异常类

定义系统中使用的各种异常类型，提供更好的错误处理和调试信息。
"""


class ChemistryLabException(Exception):
    """系统基础异常类"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        return f"[{self.error_code}] {self.message}"
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class DatabaseError(ChemistryLabException):
    """数据库相关异常"""
    
    def __init__(self, message: str, operation: str = None, table: str = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details={
                "operation": operation,
                "table": table,
            }
        )


class ModelError(ChemistryLabException):
    """模型相关异常"""
    
    def __init__(self, message: str, model_type: str = None, model_path: str = None):
        super().__init__(
            message=message,
            error_code="MODEL_ERROR", 
            details={
                "model_type": model_type,
                "model_path": model_path,
            }
        )


class APIError(ChemistryLabException):
    """API调用相关异常"""
    
    def __init__(self, message: str, api_name: str = None, status_code: int = None):
        super().__init__(
            message=message,
            error_code="API_ERROR",
            details={
                "api_name": api_name,
                "status_code": status_code,
            }
        )


class ValidationError(ChemistryLabException):
    """数据验证异常"""
    
    def __init__(self, message: str, field: str = None, value=None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={
                "field": field,
                "value": str(value) if value is not None else None,
            }
        )


class AuthenticationError(ChemistryLabException):
    """认证异常"""
    
    def __init__(self, message: str = "认证失败"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(ChemistryLabException):
    """授权异常"""
    
    def __init__(self, message: str = "权限不足"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR"
        )


class ConfigurationError(ChemistryLabException):
    """配置异常"""
    
    def __init__(self, message: str, config_key: str = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={
                "config_key": config_key,
            }
        )



class DetectionError(ChemistryLabException):
    """检测相关异常"""
    
    def __init__(self, message: str, detection_type: str = None):
        super().__init__(
            message=message,
            error_code="DETECTION_ERROR",
            details={
                "detection_type": detection_type,
            }
        )


class VideoProcessingError(ChemistryLabException):
    """视频处理异常"""
    
    def __init__(self, message: str, video_source: str = None):
        super().__init__(
            message=message,
            error_code="VIDEO_PROCESSING_ERROR",
            details={
                "video_source": video_source,
            }
        )


class LLMError(ChemistryLabException):
    """大模型相关异常"""
    
    def __init__(self, message: str, model_name: str = None):
        super().__init__(
            message=message,
            error_code="LLM_ERROR",
            details={
                "model_name": model_name,
            }
        )
