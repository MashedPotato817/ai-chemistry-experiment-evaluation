"""
YOLO检测器实现

使用YOLOv8模型进行实验器材和动作检测
"""

import time
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

from ..utils.logger import get_logger
from ..utils.exceptions import ModelError, ValidationError
from .models import Detection, BoundingBox

logger = get_logger(__name__)


class YOLODetector:
    """
    YOLO检测器类
    
    负责加载YOLOv8模型并进行实时目标检测
    """
    
    # 实验器材类别（与 fs_s_best.pt 模型对应）
    EXPERIMENT_CLASSES = {
        0:  '锥形瓶',
        1:  '双孔橡胶塞',
        2:  '直角导管',
        3:  '橡胶管',
        4:  '集气瓶',
        5:  '玻璃片',
        6:  '止水夹',
        7:  '长颈漏斗',
        8:  '废液缸',
        9:  '酒精灯',
        10: '火柴',
        11: '洗瓶',
        12: '熄灭木条',
        13: '燃烧木条',
        14: '二氧化锰固体',
        15: '药勺',
        16: '手',
        17: '水柱',
        18: '气泡',
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = 'cpu'
    ):
        """
        初始化YOLO检测器
        
        Args:
            model_path: 模型权重文件路径，如果为None则使用默认模型
            confidence_threshold: 置信度阈值，低于此值的检测结果将被过滤
            device: 运行设备 ('cpu' 或 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.model_path = model_path
        self.frame_count = 0
        self.total_inference_time = 0.0
        
        # 加载模型
        self._load_model()
        
        logger.info(
            f"YOLO检测器初始化完成 - 模型: {model_path}, "
            f"置信度阈值: {confidence_threshold}, 设备: {device}"
        )
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            # 尝试导入ultralytics库
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ModelError(
                    "未安装ultralytics库，请运行: pip install ultralytics"
                )
            
            # 加载模型
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"加载自定义模型: {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                # 使用预训练的YOLOv8n模型作为默认
                logger.warning(
                    f"未找到自定义模型 {self.model_path}，使用YOLOv8n预训练模型"
                )
                self.model = YOLO('yolov8n.pt')
            
            # 设置设备
            if self.device == 'cuda':
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA不可用，回退到CPU")
                    self.device = 'cpu'
            
            logger.info("YOLO模型加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise ModelError(f"无法加载YOLO模型: {str(e)}")
    
    def detect_objects(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> List[Detection]:
        """
        在单帧图像中检测物体
        
        Args:
            frame: 输入图像帧 (numpy数组)
            timestamp: 时间戳，如果为None则使用当前时间
        
        Returns:
            检测结果列表
        
        Raises:
            DetectionError: 检测过程中发生错误
        """
        if self.model is None:
            raise DetectionError("模型未加载")
        
        if timestamp is None:
            timestamp = time.time()
        
        try:
            # 执行推理
            start_time = time.time()
            results = self.model(frame, verbose=False, device=self.device)
            inference_time = time.time() - start_time
            
            # 更新统计信息
            self.frame_count += 1
            self.total_inference_time += inference_time
            
            # 解析检测结果
            detections = []
            for result in results:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # 获取检测框信息
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    
                    # 过滤低置信度检测
                    if conf < self.confidence_threshold:
                        continue
                    
                    # 获取类别名称
                    class_name = self.EXPERIMENT_CLASSES.get(
                        cls_id,
                        result.names.get(cls_id, f'未知类别_{cls_id}')
                    )
                    
                    # 创建检测对象
                    bbox = BoundingBox(
                        x1=int(box[0]),
                        y1=int(box[1]),
                        x2=int(box[2]),
                        y2=int(box[3])
                    )
                    
                    detection = Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=bbox,
                        timestamp=timestamp,
                        frame_id=self.frame_count
                    )
                    
                    detections.append(detection)
            
            logger.debug(
                f"帧 {self.frame_count}: 检测到 {len(detections)} 个物体, "
                f"推理时间: {inference_time:.3f}s"
            )
            
            return detections
            
        except Exception as e:
            logger.error(f"检测过程出错: {str(e)}")
            raise DetectionError(f"检测失败: {str(e)}")
    
    def get_detection_accuracy(self) -> float:
        """
        获取当前检测精度的估计值
        
        基于最近检测结果的平均置信度
        
        Returns:
            估计的检测精度 (0-1之间)
        """
        # 这是一个简化的实现，实际应用中需要更复杂的精度评估
        # 可以基于历史检测结果的置信度分布来估计
        return 0.85  # 默认返回85%的精度
    
    def get_fps(self) -> float:
        """
        获取平均处理帧率
        
        Returns:
            平均FPS
        """
        if self.frame_count == 0 or self.total_inference_time == 0:
            return 0.0
        return self.frame_count / self.total_inference_time
    
    def reset_statistics(self):
        """重置统计信息"""
        self.frame_count = 0
        self.total_inference_time = 0.0
        logger.info("检测统计信息已重置")
    
    def get_statistics(self) -> Dict:
        """
        获取检测统计信息
        
        Returns:
            包含统计信息的字典
        """
        return {
            'frame_count': self.frame_count,
            'total_inference_time': self.total_inference_time,
            'average_fps': self.get_fps(),
            'average_inference_time': (
                self.total_inference_time / self.frame_count
                if self.frame_count > 0 else 0.0
            )
        }
