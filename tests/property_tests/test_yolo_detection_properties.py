"""
YOLO检测稳定性属性测试

**特征: ai-chemistry-lab-assessment, 属性 6: YOLO检测稳定性**
**验证需求: 需求 3.1, 3.2, 3.6**

测试YOLO模型在各种输入条件下的稳定性和正确性
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck

from src.chemistry_lab.detection.yolo_detector import YOLODetector
from src.chemistry_lab.detection.models import Detection, BoundingBox
from src.chemistry_lab.utils.exceptions import DetectionError


# 测试策略：生成有效的图像尺寸
image_dimensions = st.tuples(
    st.integers(min_value=64, max_value=1920),  # 宽度
    st.integers(min_value=64, max_value=1080),  # 高度
    st.integers(min_value=1, max_value=3)  # 通道数 (灰度或RGB)
)


class TestYOLODetectionStability:
    """YOLO检测稳定性属性测试类"""
    
    @pytest.fixture(scope="class")
    def detector(self):
        """创建YOLO检测器实例"""
        # 使用较低的置信度阈值以便测试
        return YOLODetector(
            model_path=None,  # 使用默认模型
            confidence_threshold=0.3,
            device='cpu'
        )
    
    @given(image_dimensions)
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_detection_returns_valid_results(self, detector, image_dimensions):
        """
        属性: 对于任何有效的输入图像，检测器应该返回有效的检测结果列表
        
        验证:
        - 检测结果是列表类型
        - 每个检测结果都是Detection对象
        - 检测框坐标在图像范围内
        - 置信度在[0, 1]范围内
        """
        width, height, channels = image_dimensions
        
        # 生成随机图像
        if channels == 1:
            frame = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        
        # 执行检测
        detections = detector.detect_objects(frame)
        
        # 验证返回类型
        assert isinstance(detections, list), "检测结果应该是列表"
        
        # 验证每个检测结果
        for detection in detections:
            assert isinstance(detection, Detection), "每个检测结果应该是Detection对象"
            
            # 验证置信度范围
            assert 0 <= detection.confidence <= 1, \
                f"置信度应该在[0, 1]范围内: {detection.confidence}"
            
            # 验证检测框在图像范围内
            assert 0 <= detection.bbox.x1 < width, \
                f"x1坐标应该在[0, {width})范围内: {detection.bbox.x1}"
            assert 0 <= detection.bbox.y1 < height, \
                f"y1坐标应该在[0, {height})范围内: {detection.bbox.y1}"
            assert 0 < detection.bbox.x2 <= width, \
                f"x2坐标应该在(0, {width}]范围内: {detection.bbox.x2}"
            assert 0 < detection.bbox.y2 <= height, \
                f"y2坐标应该在(0, {height}]范围内: {detection.bbox.y2}"
            
            # 验证检测框的有效性
            assert detection.bbox.x1 < detection.bbox.x2, \
                "x1应该小于x2"
            assert detection.bbox.y1 < detection.bbox.y2, \
                "y1应该小于y2"
    
    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=50, deadline=None)
    def test_confidence_threshold_filtering(self, detector, threshold):
        """
        属性: 对于任何置信度阈值，所有返回的检测结果的置信度都应该大于等于该阈值
        
        验证: 检测结果的置信度 >= 设定的阈值
        """
        # 创建新的检测器实例，使用指定的阈值
        test_detector = YOLODetector(
            model_path=None,
            confidence_threshold=threshold,
            device='cpu'
        )
        
        # 生成测试图像
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # 执行检测
        detections = test_detector.detect_objects(frame)
        
        # 验证所有检测结果的置信度都大于等于阈值
        for detection in detections:
            assert detection.confidence >= threshold, \
                f"检测置信度 {detection.confidence} 应该 >= 阈值 {threshold}"
    
    def test_detection_with_black_image(self, detector):
        """
        边界情况: 纯黑图像应该返回空列表或极少检测结果
        
        验证: 系统能够处理极端输入而不崩溃
        """
        # 创建纯黑图像
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 执行检测（不应该抛出异常）
        detections = detector.detect_objects(black_frame)
        
        # 验证返回类型
        assert isinstance(detections, list), "应该返回列表"
        
        # 纯黑图像通常不会有高置信度的检测
        high_conf_detections = [d for d in detections if d.confidence > 0.7]
        assert len(high_conf_detections) == 0, \
            "纯黑图像不应该有高置信度检测"
    
    def test_detection_with_white_image(self, detector):
        """
        边界情况: 纯白图像应该返回空列表或极少检测结果
        
        验证: 系统能够处理极端输入而不崩溃
        """
        # 创建纯白图像
        white_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 执行检测（不应该抛出异常）
        detections = detector.detect_objects(white_frame)
        
        # 验证返回类型
        assert isinstance(detections, list), "应该返回列表"
    
    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_detection_consistency_across_frames(self, detector, num_frames):
        """
        属性: 对于相同的输入图像，多次检测应该返回一致的结果
        
        验证: 检测结果的数量和主要特征应该保持一致
        """
        # 生成固定的测试图像
        np.random.seed(42)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # 多次检测
        all_detections = []
        for _ in range(num_frames):
            detections = detector.detect_objects(frame)
            all_detections.append(len(detections))
        
        # 验证检测数量的一致性（允许小幅波动）
        if len(all_detections) > 1:
            detection_counts = set(all_detections)
            # 检测数量的变化应该很小（最多相差2个）
            assert max(detection_counts) - min(detection_counts) <= 2, \
                f"相同图像的检测数量变化过大: {all_detections}"
    
    def test_fps_calculation(self, detector):
        """
        测试FPS计算的正确性
        
        验证: FPS应该是正数且合理
        """
        # 重置统计
        detector.reset_statistics()
        
        # 执行几次检测
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            detector.detect_objects(frame)
        
        # 获取FPS
        fps = detector.get_fps()
        
        # 验证FPS是正数
        assert fps > 0, "FPS应该是正数"
        
        # 验证FPS在合理范围内（CPU模式下通常1-30 FPS）
        assert 0.1 <= fps <= 100, f"FPS应该在合理范围内: {fps}"
    
    def test_statistics_tracking(self, detector):
        """
        测试统计信息跟踪的正确性
        
        验证: 统计信息应该准确记录检测次数和时间
        """
        # 重置统计
        detector.reset_statistics()
        
        # 执行检测
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        num_detections = 3
        for _ in range(num_detections):
            detector.detect_objects(frame)
        
        # 获取统计信息
        stats = detector.get_statistics()
        
        # 验证统计信息
        assert stats['frame_count'] == num_detections, \
            f"帧数应该是 {num_detections}"
        assert stats['total_inference_time'] > 0, \
            "总推理时间应该大于0"
        assert stats['average_fps'] > 0, \
            "平均FPS应该大于0"
        assert stats['average_inference_time'] > 0, \
            "平均推理时间应该大于0"
    
    @given(st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=20, deadline=None)
    def test_timestamp_assignment(self, detector, timestamp):
        """
        属性: 对于任何给定的时间戳，检测结果应该正确记录该时间戳
        
        验证: 检测结果的时间戳与输入时间戳一致
        """
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # 使用指定时间戳进行检测
        detections = detector.detect_objects(frame, timestamp=timestamp)
        
        # 验证所有检测结果的时间戳
        for detection in detections:
            assert detection.timestamp == timestamp, \
                f"检测时间戳 {detection.timestamp} 应该等于输入时间戳 {timestamp}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
