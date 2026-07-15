"""
视频流处理器

负责摄像头接口、视频流管理和实时检测
"""

import time
import threading
from queue import Queue, Empty
from typing import Optional, Callable, List
import numpy as np

from ..utils.logger import get_logger
from ..utils.exceptions import VideoProcessingError
from .models import Detection
from .yolo_detector import YOLODetector

logger = get_logger(__name__)


class VideoProcessor:
    """
    视频流处理器类
    
    负责管理视频流的捕获、缓冲和实时检测
    """
    
    def __init__(
        self,
        detector: YOLODetector,
        buffer_size: int = 30,
        target_fps: int = 15
    ):
        """
        初始化视频处理器
        
        Args:
            detector: YOLO检测器实例
            buffer_size: 帧缓冲区大小
            target_fps: 目标处理帧率
        """
        self.detector = detector
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # 帧缓冲队列
        self.frame_queue = Queue(maxsize=buffer_size)
        self.detection_queue = Queue(maxsize=buffer_size)
        
        # 控制标志
        self.is_running = False
        self.capture_thread = None
        self.detection_thread = None
        
        # 摄像头对象
        self.camera = None
        self.camera_id = 0
        
        logger.info(
            f"视频处理器初始化 - 缓冲区: {buffer_size}, "
            f"目标FPS: {target_fps}"
        )

    def start_camera(self, camera_id: int = 0) -> bool:
        """
        启动摄像头
        
        Args:
            camera_id: 摄像头设备ID
        
        Returns:
            是否成功启动
        """
        try:
            import cv2
            
            self.camera_id = camera_id
            self.camera = cv2.VideoCapture(camera_id)
            
            if not self.camera.isOpened():
                raise VideoProcessingError(f"无法打开摄像头 {camera_id}")
            
            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            logger.info(f"摄像头 {camera_id} 启动成功")
            return True
            
        except Exception as e:
            logger.error(f"启动摄像头失败: {str(e)}")
            raise VideoProcessingError(f"摄像头启动失败: {str(e)}")
    
    def stop_camera(self):
        """停止摄像头"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            logger.info("摄像头已停止")
    
    def _capture_loop(self):
        """摄像头捕获循环（在独立线程中运行）"""
        logger.info("开始视频捕获循环")
        last_capture_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # 控制帧率
                if current_time - last_capture_time < self.frame_interval:
                    time.sleep(0.001)
                    continue

                # 读取帧
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("无法读取视频帧")
                    continue
                
                # 添加时间戳
                timestamp = current_time
                
                # 将帧放入队列（非阻塞）
                try:
                    self.frame_queue.put_nowait((frame, timestamp))
                    last_capture_time = current_time
                except:
                    # 队列已满，丢弃最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((frame, timestamp))
                    except:
                        pass
                        
            except Exception as e:
                logger.error(f"捕获循环错误: {str(e)}")
                time.sleep(0.1)
        
        logger.info("视频捕获循环结束")
    
    def _detection_loop(self):
        """检测处理循环（在独立线程中运行）"""
        logger.info("开始检测处理循环")
        
        while self.is_running:
            try:
                # 从队列获取帧
                frame, timestamp = self.frame_queue.get(timeout=1.0)
                
                # 执行检测
                detections = self.detector.detect_objects(frame, timestamp)
                
                # 将检测结果放入队列
                try:
                    self.detection_queue.put_nowait({
                        'frame': frame,
                        'timestamp': timestamp,
                        'detections': detections
                    })
                except:
                    # 队列已满，丢弃最旧的结果
                    try:
                        self.detection_queue.get_nowait()
                        self.detection_queue.put_nowait({
                            'frame': frame,
                            'timestamp': timestamp,
                            'detections': detections
                        })
                    except:
                        pass
                        
            except Empty:
                continue
            except Exception as e:
                logger.error(f"检测循环错误: {str(e)}")
                time.sleep(0.1)
        
        logger.info("检测处理循环结束")

    def start_processing(self):
        """启动视频处理"""
        if self.is_running:
            logger.warning("视频处理已在运行")
            return
        
        if self.camera is None:
            raise VideoProcessingError("摄像头未启动")
        
        self.is_running = True
        
        # 启动捕获线程
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self.capture_thread.start()
        
        # 启动检测线程
        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True
        )
        self.detection_thread.start()
        
        logger.info("视频处理已启动")
    
    def stop_processing(self):
        """停止视频处理"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 等待线程结束
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        while not self.detection_queue.empty():
            try:
                self.detection_queue.get_nowait()
            except:
                break
        
        logger.info("视频处理已停止")
    
    def get_latest_detection(self, timeout: float = 1.0) -> Optional[dict]:
        """
        获取最新的检测结果
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            包含帧、时间戳和检测结果的字典，如果超时则返回None
        """
        try:
            return self.detection_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        处理单个帧（同步方法）
        
        Args:
            frame: 输入帧
        
        Returns:
            检测结果列表
        """
        return self.detector.detect_objects(frame)
    
    def __del__(self):
        """析构函数，确保资源释放"""
        self.stop_processing()
        self.stop_camera()
