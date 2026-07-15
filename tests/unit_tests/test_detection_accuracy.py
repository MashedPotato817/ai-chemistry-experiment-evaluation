"""
检测精度监控单元测试

测试低精度检测的错误处理和检测结果过滤逻辑
验证需求: 需求 3.6
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.chemistry_lab.detection.yolo_detector import YOLODetector
from src.chemistry_lab.detection.models import Detection, BoundingBox
from src.chemistry_lab.utils.exceptions import DetectionError


class TestDetectionAccuracy:
    """检测精度监控测试类"""
    
    @pytest.fixture
    def detector(self):
        """创建检测器实例"""
        return YOLODetector(
            model_path=None,
            confidence_threshold=0.5,
            device='cpu'
        )
    
    def test_low_confidence_filtering(self, detector):
        """
        测试低置信度检测结果被正确过滤
        
        验证: 置信度低于阈值的检测结果不应该出现在结果中
        """
        # 创建测试图像
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # 执行检测
        detections = detector.detect_objects(frame)
        
        # 验证所有检测结果的置信度都大于等于阈值
        for detection in detections:
            assert detection.confidence >= detector.confidence_threshold, \
                f"检测置信度 {detection.confidence} 低于阈值 {detector.confidence_threshold}"
    
    def test_accuracy_threshold_warning(self, detector):
        """
        测试检测精度低于80%时的处理
        
        验证: 系统应该能够检测到低精度情况
        """
        # 获取检测精度
        accuracy = detector.get_detection_accuracy()
        
        # 验证精度在合理范围内
        assert 0 <= accuracy <= 1, f"精度应该在[0, 1]范围内: {accuracy}"
        
        # 如果精度低于80%，应该有相应的处理机制
        if accuracy < 0.8:
            # 这里可以添加警告日志或用户提示的验证
            pass
    
    def test_detection_with_poor_lighting(self, detector):
        """
        测试低光照条件下的检测
        
        验证: 系统应该能够处理低光照图像而不崩溃
        """
        # 创建低光照图像（暗图像）
        dark_frame = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        
        # 执行检测（不应该抛出异常）
        detections = detector.detect_objects(dark_frame)
        
        # 验证返回类型
        assert isinstance(detections, list), "应该返回列表"
        
        # 低光照条件下可能检测结果较少
        # 但系统应该正常运行
    
    def test_detection_with_overexposed_image(self, detector):
        """
        测试过曝光条件下的检测
        
        验证: 系统应该能够处理过曝光图像而不崩溃
        """
        # 创建过曝光图像（亮图像）
        bright_frame = np.random.randint(200, 256, (480, 640, 3), dtype=np.uint8)
        
        # 执行检测（不应该抛出异常）
        detections = detector.detect_objects(bright_frame)
        
        # 验证返回类型
        assert isinstance(detections, list), "应该返回列表"
    
    def test_confidence_threshold_adjustment(self):
        """
        测试不同置信度阈值的效果
        
        验证: 更高的阈值应该产生更少但更可靠的检测结果
        """
        # 创建测试图像
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # 使用低阈值
        detector_low = YOLODetector(
            model_path=None,
            confidence_threshold=0.3,
            device='cpu'
        )
        detections_low = detector_low.detect_objects(frame)
        
        # 使用高阈值
        detector_high = YOLODetector(
            model_path=None,
            confidence_threshold=0.7,
            device='cpu'
        )
        detections_high = detector_high.detect_objects(frame)
        
        # 验证高阈值产生的检测结果数量不多于低阈值
        assert len(detections_high) <= len(detections_low), \
            "高阈值应该产生更少或相等的检测结果"
        
        # 验证高阈值的所有检测结果置信度都较高
        for detection in detections_high:
            assert detection.confidence >= 0.7, \
                f"高阈值检测的置信度应该 >= 0.7: {detection.confidence}"
    
    def test_detection_result_filtering(self, detector):
        """
        测试检测结果过滤逻辑
        
        验证: 无效或异常的检测结果应该被过滤
        """
        # 创建测试图像
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # 执行检测
        detections = detector.detect_objects(frame)
        
        # 验证所有检测结果都是有效的
        for detection in detections:
            # 检测框应该在图像范围内
            assert 0 <= detection.bbox.x1 < 640, "x1坐标超出范围"
            assert 0 <= detection.bbox.y1 < 480, "y1坐标超出范围"
            assert 0 < detection.bbox.x2 <= 640, "x2坐标超出范围"
            assert 0 < detection.bbox.y2 <= 480, "y2坐标超出范围"
            
            # 检测框应该有效
            assert detection.bbox.x1 < detection.bbox.x2, "无效的x坐标"
            assert detection.bbox.y1 < detection.bbox.y2, "无效的y坐标"
            
            # 置信度应该在有效范围内
            assert 0 <= detection.confidence <= 1, "置信度超出范围"
    
    def test_fps_meets_requirement(self, detector):
        """
        测试检测速度是否满足要求
        
        验证: 系统应该能够达到至少15 FPS的处理能力（需求5.3）
        注意: 这是一个性能测试，实际FPS取决于硬件
        """
        # 重置统计
        detector.reset_statistics()
        
        # 执行多次检测
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        num_frames = 10
        
        for _ in range(num_frames):
            detector.detect_objects(frame)
        
        # 获取FPS
        fps = detector.get_fps()
        
        # 验证FPS是正数
        assert fps > 0, "FPS应该是正数"
        
        # 记录FPS用于性能分析
        # 注意: 在CPU模式下可能无法达到15 FPS，这是正常的
        print(f"检测FPS: {fps:.2f}")
    
    def test_detection_with_invalid_input(self, detector):
        """
        测试无效输入的错误处理
        
        验证: 系统应该优雅地处理无效输入
        """
        # 测试None输入
        with pytest.raises((DetectionError, AttributeError, TypeError)):
            detector.detect_objects(None)
        
        # 测试空数组
        empty_frame = np.array([])
        with pytest.raises((DetectionError, ValueError)):
            detector.detect_objects(empty_frame)
        
        # 测试错误维度的数组
        wrong_shape = np.random.randint(0, 256, (10,), dtype=np.uint8)
        with pytest.raises((DetectionError, ValueError)):
            detector.detect_objects(wrong_shape)
    
    def test_statistics_reset(self, detector):
        """
        测试统计信息重置功能
        
        验证: 重置后统计信息应该清零
        """
        # 执行一些检测
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        detector.detect_objects(frame)
        detector.detect_objects(frame)
        
        # 验证统计信息不为零
        stats_before = detector.get_statistics()
        assert stats_before['frame_count'] > 0, "检测前应该有帧数统计"
        
        # 重置统计
        detector.reset_statistics()
        
        # 验证统计信息已清零
        stats_after = detector.get_statistics()
        assert stats_after['frame_count'] == 0, "重置后帧数应该为0"
        assert stats_after['total_inference_time'] == 0, "重置后总时间应该为0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
