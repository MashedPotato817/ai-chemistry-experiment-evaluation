"""
时序检测数据完整性属性测试

**特征: ai-chemistry-lab-assessment, 属性 3: 时序检测数据完整性**
**验证需求: 需求 3.4, 需求 8.3**

测试时序检测数据的完整性和时间逻辑正确性
"""

import pytest
import time
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck

from src.chemistry_lab.detection.models import (
    Detection, BoundingBox, Action, ExperimentStep
)
from src.chemistry_lab.detection.temporal_analyzer import TemporalAnalyzer


# 测试策略
timestamps = st.floats(min_value=0.0, max_value=1000000.0, allow_nan=False, allow_infinity=False)
positive_floats = st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False)
step_names = st.sampled_from(['取样', '加液', '搅拌', '加热', '观察', '记录'])


class TestTemporalDetectionIntegrity:
    """时序检测数据完整性属性测试类"""
    
    @given(
        st.floats(min_value=0.0, max_value=1000.0),
        st.floats(min_value=0.1, max_value=100.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_action_time_ordering(self, start_time, duration):
        """
        属性: 对于任何动作，开始时间必须小于结束时间
        
        验证: start_time < end_time
        """
        end_time = start_time + duration
        
        action = Action(
            action_type='测试动作',
            start_time=start_time,
            end_time=end_time,
            involved_objects=['物体1'],
            confidence=0.8
        )
        
        assert action.start_time < action.end_time, \
            f"开始时间 {action.start_time} 应该小于结束时间 {action.end_time}"
        
        # 验证持续时间计算正确
        assert action.duration == duration, \
            f"持续时间应该是 {duration}，实际是 {action.duration}"
    
    @given(
        st.floats(min_value=0.0, max_value=1000.0),
        st.floats(min_value=0.1, max_value=100.0),
        st.floats(min_value=0.0, max_value=50.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_experiment_step_time_ordering(self, start_time, duration, pause):
        """
        属性: 对于任何实验步骤，时间戳应该满足逻辑顺序
        
        验证: start_timestamp < end_timestamp 且 duration >= 0 且 pause_duration >= 0
        """
        end_time = start_time + duration
        
        step = ExperimentStep(
            step_name='测试步骤',
            start_timestamp=start_time,
            end_timestamp=end_time,
            duration=duration,
            pause_duration=pause,
            anomalies=[],
            sequence_order=1,
            detected_objects=['物体1']
        )
        
        # 验证时间顺序
        assert step.start_timestamp < step.end_timestamp, \
            f"开始时间戳 {step.start_timestamp} 应该小于结束时间戳 {step.end_timestamp}"
        
        # 验证持续时间非负
        assert step.duration >= 0, \
            f"持续时间应该非负: {step.duration}"
        
        # 验证停顿时间非负
        assert step.pause_duration >= 0, \
            f"停顿时间应该非负: {step.pause_duration}"
    
    @given(st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1000.0),
            st.floats(min_value=0.1, max_value=10.0)
        ),
        min_size=2,
        max_size=10
    ))
    @settings(max_examples=50, deadline=None)
    def test_step_sequence_ordering(self, time_pairs):
        """
        属性: 对于任何实验步骤序列，步骤应该按时间顺序排列
        
        验证: 对于序列中的任意相邻步骤i和i+1，step[i].start_timestamp <= step[i+1].start_timestamp
        """
        # 按开始时间排序
        sorted_pairs = sorted(time_pairs, key=lambda x: x[0])
        
        # 创建步骤列表
        steps = []
        for i, (start_time, duration) in enumerate(sorted_pairs):
            step = ExperimentStep(
                step_name=f'步骤{i+1}',
                start_timestamp=start_time,
                end_timestamp=start_time + duration,
                duration=duration,
                pause_duration=0.0,
                anomalies=[],
                sequence_order=i + 1,
                detected_objects=[]
            )
            steps.append(step)
        
        # 验证序列顺序
        for i in range(len(steps) - 1):
            assert steps[i].start_timestamp <= steps[i+1].start_timestamp, \
                f"步骤 {i} 的开始时间 {steps[i].start_timestamp} 应该 <= " \
                f"步骤 {i+1} 的开始时间 {steps[i+1].start_timestamp}"
    
    @given(
        st.integers(min_value=0, max_value=1920),
        st.integers(min_value=0, max_value=1080),
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_bounding_box_validity(self, x1, y1, width, height):
        """
        属性: 对于任何检测框，坐标应该满足 x1 < x2 且 y1 < y2
        
        验证: 检测框的几何有效性
        """
        assume(width > 0 and height > 0)
        assume(x1 + width <= 1920 and y1 + height <= 1080)
        
        x2 = x1 + width
        y2 = y1 + height
        
        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        
        # 验证坐标顺序
        assert bbox.x1 < bbox.x2, f"x1 {bbox.x1} 应该小于 x2 {bbox.x2}"
        assert bbox.y1 < bbox.y2, f"y1 {bbox.y1} 应该小于 y2 {bbox.y2}"
        
        # 验证尺寸计算
        assert bbox.width == width, f"宽度应该是 {width}"
        assert bbox.height == height, f"高度应该是 {height}"
        
        # 验证面积计算
        expected_area = width * height
        assert bbox.area == expected_area, f"面积应该是 {expected_area}"
    
    def test_detection_timestamp_consistency(self):
        """
        测试检测结果的时间戳一致性
        
        验证: 同一帧的所有检测结果应该有相同的时间戳
        """
        timestamp = time.time()
        frame_id = 1
        
        detections = [
            Detection(
                class_id=i,
                class_name=f'物体{i}',
                confidence=0.8,
                bbox=BoundingBox(x1=10*i, y1=10*i, x2=50*i, y2=50*i),
                timestamp=timestamp,
                frame_id=frame_id
            )
            for i in range(1, 5)
        ]
        
        # 验证所有检测的时间戳相同
        for detection in detections:
            assert detection.timestamp == timestamp, \
                f"检测时间戳应该是 {timestamp}"
            assert detection.frame_id == frame_id, \
                f"帧ID应该是 {frame_id}"
    
    @given(st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=100),
            st.floats(min_value=0.0, max_value=100.0)
        ),
        min_size=1,
        max_size=20
    ))
    @settings(max_examples=50, deadline=None)
    def test_temporal_analyzer_tracking(self, detections_data):
        """
        属性: 时序分析器应该正确追踪所有检测到的物体
        
        验证: 添加的检测结果应该被正确记录和追踪
        """
        analyzer = TemporalAnalyzer(history_size=30)
        
        # 创建检测结果
        for frame_id, (class_id, timestamp) in enumerate(detections_data):
            detections = [
                Detection(
                    class_id=class_id,
                    class_name=f'物体{class_id}',
                    confidence=0.8,
                    bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
                    timestamp=timestamp,
                    frame_id=frame_id
                )
            ]
            analyzer.add_detections(detections)
        
        # 验证历史记录
        stats = analyzer.get_statistics()
        assert stats['history_length'] <= len(detections_data), \
            "历史记录长度不应超过添加的检测数量"
        assert stats['tracked_objects'] > 0, \
            "应该至少追踪到一个物体"
    
    def test_action_duration_calculation(self):
        """
        测试动作持续时间计算的正确性
        
        验证: duration = end_time - start_time
        """
        start_time = 10.0
        end_time = 15.5
        expected_duration = 5.5
        
        action = Action(
            action_type='测试',
            start_time=start_time,
            end_time=end_time,
            involved_objects=['物体1'],
            confidence=0.9
        )
        
        assert abs(action.duration - expected_duration) < 0.001, \
            f"持续时间应该是 {expected_duration}，实际是 {action.duration}"
    
    def test_invalid_time_ordering_raises_error(self):
        """
        测试无效的时间顺序应该抛出错误
        
        验证: start_time >= end_time 应该抛出ValueError
        """
        # 测试Action
        with pytest.raises(ValueError):
            Action(
                action_type='无效动作',
                start_time=10.0,
                end_time=5.0,  # 结束时间早于开始时间
                involved_objects=['物体1'],
                confidence=0.8
            )
        
        # 测试ExperimentStep
        with pytest.raises(ValueError):
            ExperimentStep(
                step_name='无效步骤',
                start_timestamp=10.0,
                end_timestamp=5.0,  # 结束时间早于开始时间
                duration=5.0,
                pause_duration=0.0,
                anomalies=[],
                sequence_order=1,
                detected_objects=[]
            )
    
    def test_negative_duration_raises_error(self):
        """
        测试负数持续时间应该抛出错误
        
        验证: duration < 0 或 pause_duration < 0 应该抛出ValueError
        """
        with pytest.raises(ValueError):
            ExperimentStep(
                step_name='无效步骤',
                start_timestamp=0.0,
                end_timestamp=10.0,
                duration=-5.0,  # 负数持续时间
                pause_duration=0.0,
                anomalies=[],
                sequence_order=1,
                detected_objects=[]
            )
        
        with pytest.raises(ValueError):
            ExperimentStep(
                step_name='无效步骤',
                start_timestamp=0.0,
                end_timestamp=10.0,
                duration=10.0,
                pause_duration=-2.0,  # 负数停顿时间
                anomalies=[],
                sequence_order=1,
                detected_objects=[]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
