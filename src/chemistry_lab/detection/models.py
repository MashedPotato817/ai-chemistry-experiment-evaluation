"""
检测相关的数据模型
"""

from dataclasses import dataclass
from typing import Tuple
from datetime import datetime


@dataclass
class BoundingBox:
    """检测框坐标"""
    x1: int  # 左上角x坐标
    y1: int  # 左上角y坐标
    x2: int  # 右下角x坐标
    y2: int  # 右下角y坐标
    
    @property
    def center(self) -> Tuple[float, float]:
        """返回检测框中心点坐标"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        """返回检测框面积"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    @property
    def width(self) -> int:
        """返回检测框宽度"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """返回检测框高度"""
        return self.y2 - self.y1


@dataclass
class Detection:
    """单次检测结果"""
    class_id: int  # 类别ID
    class_name: str  # 类别名称
    confidence: float  # 置信度
    bbox: BoundingBox  # 检测框
    timestamp: float  # 时间戳
    frame_id: int  # 帧ID
    
    def __post_init__(self):
        """验证检测结果的有效性"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"置信度必须在[0, 1]范围内，当前值: {self.confidence}")
        if self.bbox.x1 >= self.bbox.x2 or self.bbox.y1 >= self.bbox.y2:
            raise ValueError(f"无效的检测框坐标: {self.bbox}")


@dataclass
class Action:
    """实验动作"""
    action_type: str  # 动作类型（如：取样、加液、搅拌）
    start_time: float  # 开始时间
    end_time: float  # 结束时间
    involved_objects: list  # 涉及的物体列表
    confidence: float  # 置信度
    
    @property
    def duration(self) -> float:
        """返回动作持续时间"""
        return self.end_time - self.start_time
    
    def __post_init__(self):
        """验证动作数据的有效性"""
        if self.start_time >= self.end_time:
            raise ValueError(f"开始时间必须小于结束时间: start={self.start_time}, end={self.end_time}")


@dataclass
class ExperimentStep:
    """实验步骤"""
    step_name: str  # 步骤名称
    start_timestamp: float  # 开始时间戳
    end_timestamp: float  # 结束时间戳
    duration: float  # 持续时间
    pause_duration: float  # 停顿时间
    anomalies: list  # 异常操作列表
    sequence_order: int  # 步骤顺序
    detected_objects: list  # 检测到的物体列表
    
    def __post_init__(self):
        """验证步骤数据的有效性"""
        if self.start_timestamp >= self.end_timestamp:
            raise ValueError(
                f"开始时间戳必须小于结束时间戳: "
                f"start={self.start_timestamp}, end={self.end_timestamp}"
            )
        if self.duration < 0:
            raise ValueError(f"持续时间不能为负数: {self.duration}")
        if self.pause_duration < 0:
            raise ValueError(f"停顿时间不能为负数: {self.pause_duration}")
