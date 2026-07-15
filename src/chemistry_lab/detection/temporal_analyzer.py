"""
时序分析器

分析检测框的时序变化，识别实验动作和步骤。
针对气体制备与收集实验（fs_s_best.pt 模型）优化。

模型可识别的19个类别：
  锥形瓶、双孔橡胶塞、直角导管、橡胶管、集气瓶、玻璃片、
  止水夹、长颈漏斗、废液缸、酒精灯、火柴、洗瓶、
  熄灭木条、燃烧木条、二氧化锰固体、药勺、手、水柱、气泡
"""

import time
from typing import List, Dict, Optional, Tuple
from collections import deque
import numpy as np

from ..utils.logger import get_logger
from .models import Detection, Action, ExperimentStep

logger = get_logger(__name__)


# ── 实验步骤定义 ──────────────────────────────────────────────────────────────
# 气体制备与收集实验的标准步骤序列
EXPERIMENT_STEPS_SEQUENCE = [
    "准备器材",       # 锥形瓶、双孔橡胶塞、导管等出现在画面中
    "添加药品",       # 手+药勺+二氧化锰固体+锥形瓶
    "组装装置",       # 手操作橡胶塞、导管、橡胶管、止水夹
    "检查气密性",     # 手+止水夹+水柱（气密性检验）
    "点燃酒精灯",     # 手+火柴+酒精灯
    "收集气体",       # 气泡+集气瓶+水柱
    "验证气体",       # 手+燃烧木条/熄灭木条+集气瓶
    "熄灭酒精灯",     # 手+酒精灯（停止加热）
    "整理器材",       # 废液缸+洗瓶
]

# 安全相关器材（出现异常操作时需要警告）
SAFETY_CRITICAL_OBJECTS = {"酒精灯", "火柴", "燃烧木条"}

# 各步骤的关键器材组合（用于步骤识别）
STEP_INDICATORS: Dict[str, Dict] = {
    "准备器材": {
        "required": {"锥形瓶"},
        "optional": {"双孔橡胶塞", "直角导管", "橡胶管", "长颈漏斗", "集气瓶"},
    },
    "添加药品": {
        "required": {"手", "药勺"},
        "optional": {"二氧化锰固体", "锥形瓶"},
    },
    "组装装置": {
        "required": {"手"},
        "optional": {"双孔橡胶塞", "直角导管", "橡胶管", "止水夹"},
    },
    "检查气密性": {
        "required": {"止水夹"},
        "optional": {"水柱", "手"},
    },
    "点燃酒精灯": {
        "required": {"酒精灯", "火柴"},
        "optional": {"手"},
    },
    "收集气体": {
        "required": {"集气瓶"},
        "optional": {"气泡", "水柱", "玻璃片"},
    },
    "验证气体": {
        "required": {"集气瓶"},
        "optional": {"燃烧木条", "熄灭木条", "手"},
    },
    "熄灭酒精灯": {
        "required": {"酒精灯"},
        "optional": {"手"},
    },
    "整理器材": {
        "required": set(),
        "optional": {"废液缸", "洗瓶", "手"},
    },
}


class TemporalAnalyzer:
    """
    时序分析器

    负责分析检测结果的时序变化，识别动作和实验步骤。
    针对气体制备与收集实验优化。
    """

    def __init__(
        self,
        history_size: int = 60,
        movement_threshold: float = 40.0,
        proximity_threshold: float = 120.0,
    ):
        """
        初始化时序分析器

        Args:
            history_size: 保留的历史帧数（60帧 ≈ 1秒@60fps）
            movement_threshold: 判定为移动的最小距离（像素）
            proximity_threshold: 判定为接近的最大距离（像素）
        """
        self.history_size = history_size
        self.movement_threshold = movement_threshold
        self.proximity_threshold = proximity_threshold

        # 历史检测结果
        self.detection_history: deque = deque(maxlen=history_size)

        # 物体追踪字典 {class_name: deque([{center, timestamp, bbox}])}
        self.object_tracks: Dict[str, deque] = {}

        # 识别的动作列表
        self.identified_actions: List[Action] = []

        # 当前推断的实验步骤
        self.current_step: str = "准备器材"
        self.step_start_time: float = time.time()

        # 安全事件记录
        self.safety_events: List[Dict] = []

        logger.info(
            f"时序分析器初始化 - 历史帧数: {history_size}, "
            f"移动阈值: {movement_threshold}px, 接近阈值: {proximity_threshold}px"
        )

    # ── 核心数据更新 ──────────────────────────────────────────────────────────

    def add_detections(self, detections: List[Detection]) -> None:
        """
        添加新的检测结果到历史记录，并触发分析

        Args:
            detections: 当前帧的检测结果列表
        """
        if not detections:
            return

        self.detection_history.append(detections)
        self._update_object_tracks(detections)

        # 实时推断当前步骤
        detected_names = {d.class_name for d in detections}
        self._infer_current_step(detected_names)

        # 安全检查
        self._check_safety_events(detections)

    def _update_object_tracks(self, detections: List[Detection]) -> None:
        """更新物体追踪信息"""
        for detection in detections:
            name = detection.class_name
            if name not in self.object_tracks:
                self.object_tracks[name] = deque(maxlen=self.history_size)
            self.object_tracks[name].append({
                "center": detection.bbox.center,
                "timestamp": detection.timestamp,
                "bbox": detection.bbox,
                "confidence": detection.confidence,
            })

    # ── 步骤推断 ──────────────────────────────────────────────────────────────

    def _infer_current_step(self, detected_names: set) -> None:
        """
        根据当前帧检测到的器材推断实验步骤

        Args:
            detected_names: 当前帧检测到的类别名称集合
        """
        best_step = self.current_step
        best_score = -1

        for step, indicators in STEP_INDICATORS.items():
            required = indicators["required"]
            optional = indicators["optional"]

            # 必须器材全部出现才计分
            if required and not required.issubset(detected_names):
                continue

            score = len(required) * 2 + len(optional & detected_names)
            if score > best_score:
                best_score = score
                best_step = step

        if best_step != self.current_step:
            logger.info(f"实验步骤切换: {self.current_step} → {best_step}")
            self.current_step = best_step
            self.step_start_time = time.time()

    def get_current_step(self) -> str:
        """返回当前推断的实验步骤"""
        return self.current_step

    def get_step_elapsed_time(self) -> float:
        """返回当前步骤已用时间（秒）"""
        return time.time() - self.step_start_time

    # ── 安全检查 ──────────────────────────────────────────────────────────────

    def _check_safety_events(self, detections: List[Detection]) -> None:
        """
        检查安全相关事件

        当酒精灯、火柴、燃烧木条出现时，检查手是否保持安全距离
        """
        detected_names = {d.class_name for d in detections}
        safety_objects_present = SAFETY_CRITICAL_OBJECTS & detected_names

        if not safety_objects_present or "手" not in detected_names:
            return

        # 检查手与危险器材的距离
        hand_tracks = self.object_tracks.get("手")
        if not hand_tracks:
            return

        hand_pos = hand_tracks[-1]["center"]

        for obj_name in safety_objects_present:
            obj_tracks = self.object_tracks.get(obj_name)
            if not obj_tracks:
                continue

            obj_pos = obj_tracks[-1]["center"]
            distance = np.sqrt(
                (hand_pos[0] - obj_pos[0]) ** 2 + (hand_pos[1] - obj_pos[1]) ** 2
            )

            # 手距离危险器材过近（< 60px）时记录安全事件
            if distance < 60:
                event = {
                    "timestamp": time.time(),
                    "type": "proximity_warning",
                    "object": obj_name,
                    "distance": distance,
                    "message": f"手距离{obj_name}过近（{distance:.0f}px），注意安全",
                }
                self.safety_events.append(event)
                logger.warning(event["message"])

    def get_recent_safety_events(self, seconds: float = 5.0) -> List[Dict]:
        """
        获取最近的安全事件

        Args:
            seconds: 时间窗口（秒）

        Returns:
            安全事件列表
        """
        cutoff = time.time() - seconds
        return [e for e in self.safety_events if e["timestamp"] >= cutoff]

    # ── 运动分析 ──────────────────────────────────────────────────────────────

    def detect_movement(
        self,
        object_name: str,
        time_window: float = 1.0,
    ) -> Optional[Tuple[float, float]]:
        """
        检测物体的移动向量

        Args:
            object_name: 物体名称
            time_window: 时间窗口（秒）

        Returns:
            移动向量 (dx, dy)，未移动则返回 None
        """
        if object_name not in self.object_tracks:
            return None

        tracks = list(self.object_tracks[object_name])
        if len(tracks) < 2:
            return None

        current_time = tracks[-1]["timestamp"]
        window_tracks = [t for t in tracks if current_time - t["timestamp"] <= time_window]

        if len(window_tracks) < 2:
            return None

        start_pos = window_tracks[0]["center"]
        end_pos = window_tracks[-1]["center"]
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]

        if np.sqrt(dx ** 2 + dy ** 2) >= self.movement_threshold:
            return (dx, dy)
        return None

    def detect_proximity(self, object1_name: str, object2_name: str) -> bool:
        """
        检测两个物体是否接近

        Args:
            object1_name: 第一个物体名称
            object2_name: 第二个物体名称

        Returns:
            是否接近
        """
        t1 = self.object_tracks.get(object1_name)
        t2 = self.object_tracks.get(object2_name)
        if not t1 or not t2:
            return False

        pos1 = t1[-1]["center"]
        pos2 = t2[-1]["center"]
        distance = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        return distance <= self.proximity_threshold

    def get_object_velocity(
        self,
        object_name: str,
        time_window: float = 0.5,
    ) -> Optional[float]:
        """
        计算物体的移动速度（像素/秒）

        Args:
            object_name: 物体名称
            time_window: 时间窗口（秒）

        Returns:
            速度，无法计算则返回 None
        """
        if object_name not in self.object_tracks:
            return None

        tracks = list(self.object_tracks[object_name])
        if len(tracks) < 2:
            return None

        current_time = tracks[-1]["timestamp"]
        window_tracks = [t for t in tracks if current_time - t["timestamp"] <= time_window]

        if len(window_tracks) < 2:
            return None

        total_distance = sum(
            np.sqrt(
                (window_tracks[i]["center"][0] - window_tracks[i - 1]["center"][0]) ** 2
                + (window_tracks[i]["center"][1] - window_tracks[i - 1]["center"][1]) ** 2
            )
            for i in range(1, len(window_tracks))
        )

        total_time = window_tracks[-1]["timestamp"] - window_tracks[0]["timestamp"]
        return total_distance / total_time if total_time > 0 else 0.0

    def detect_pause(
        self,
        object_name: str,
        pause_threshold: float = 5.0,
        time_window: float = 2.0,
    ) -> bool:
        """
        检测物体是否处于停顿状态

        Args:
            object_name: 物体名称
            pause_threshold: 停顿速度阈值（像素/秒）
            time_window: 检测时间窗口（秒）

        Returns:
            是否处于停顿状态
        """
        velocity = self.get_object_velocity(object_name, time_window)
        if velocity is None:
            return False
        return velocity < pause_threshold

    # ── 动作序列分析 ──────────────────────────────────────────────────────────

    def analyze_action_sequence(self, time_window: float = 5.0) -> List[Action]:
        """
        分析时间窗口内的动作序列

        基于手部运动和器材接近关系识别操作动作。
        针对气体制备实验的器材组合进行优化识别。

        Args:
            time_window: 分析的时间窗口（秒）

        Returns:
            识别的动作列表
        """
        actions = []

        if len(self.detection_history) < 2:
            return actions

        # 手部动作分析
        if "手" in self.object_tracks:
            hand_movement = self.detect_movement("手", time_window)

            if hand_movement:
                # 检查手接近的器材，生成操作动作
                for obj_name in list(self.object_tracks.keys()):
                    if obj_name == "手":
                        continue
                    if self.detect_proximity("手", obj_name):
                        action = self._create_action(
                            action_type=self._classify_action(obj_name),
                            involved_objects=["手", obj_name],
                            time_window=time_window,
                        )
                        if action:
                            actions.append(action)

        # 特殊事件：气泡出现（产气中）
        if "气泡" in self.object_tracks and "集气瓶" in self.object_tracks:
            if self.detect_proximity("气泡", "集气瓶"):
                action = self._create_action(
                    action_type="收集气体",
                    involved_objects=["气泡", "集气瓶"],
                    time_window=time_window,
                )
                if action:
                    actions.append(action)

        # 特殊事件：燃烧木条验证
        if "燃烧木条" in self.object_tracks and "集气瓶" in self.object_tracks:
            if self.detect_proximity("燃烧木条", "集气瓶"):
                action = self._create_action(
                    action_type="验证气体（燃烧木条）",
                    involved_objects=["燃烧木条", "集气瓶"],
                    time_window=time_window,
                )
                if action:
                    actions.append(action)

        # 特殊事件：熄灭木条验证
        if "熄灭木条" in self.object_tracks and "集气瓶" in self.object_tracks:
            if self.detect_proximity("熄灭木条", "集气瓶"):
                action = self._create_action(
                    action_type="验证气体（熄灭木条）",
                    involved_objects=["熄灭木条", "集气瓶"],
                    time_window=time_window,
                )
                if action:
                    actions.append(action)

        # 记录到历史
        self.identified_actions.extend(actions)
        return actions

    def _classify_action(self, obj_name: str) -> str:
        """
        根据器材名称推断操作类型

        Args:
            obj_name: 器材名称

        Returns:
            操作类型描述
        """
        action_map = {
            "锥形瓶":     "操作锥形瓶",
            "双孔橡胶塞": "安装橡胶塞",
            "直角导管":   "连接导管",
            "橡胶管":     "连接橡胶管",
            "集气瓶":     "操作集气瓶",
            "玻璃片":     "放置玻璃片",
            "止水夹":     "操作止水夹",
            "长颈漏斗":   "使用长颈漏斗",
            "废液缸":     "倒入废液缸",
            "酒精灯":     "操作酒精灯",
            "火柴":       "使用火柴",
            "洗瓶":       "使用洗瓶",
            "熄灭木条":   "验证气体（熄灭木条）",
            "燃烧木条":   "验证气体（燃烧木条）",
            "二氧化锰固体": "添加催化剂",
            "药勺":       "使用药勺取药",
            "水柱":       "观察水柱",
            "气泡":       "观察气泡",
        }
        return action_map.get(obj_name, f"操作{obj_name}")

    def _create_action(
        self,
        action_type: str,
        involved_objects: List[str],
        time_window: float,
    ) -> Optional[Action]:
        """创建动作对象"""
        if not involved_objects:
            return None

        current_time = time.time()
        return Action(
            action_type=action_type,
            start_time=current_time - time_window,
            end_time=current_time,
            involved_objects=involved_objects,
            confidence=0.8,
        )

    # ── 器材完整性检查 ────────────────────────────────────────────────────────

    def check_apparatus_completeness(self) -> Dict[str, bool]:
        """
        检查实验所需器材是否齐全

        Returns:
            {器材名称: 是否已检测到}
        """
        required_apparatus = [
            "锥形瓶", "双孔橡胶塞", "直角导管", "橡胶管",
            "集气瓶", "止水夹", "长颈漏斗", "酒精灯",
            "二氧化锰固体", "药勺",
        ]
        return {name: name in self.object_tracks for name in required_apparatus}

    def get_missing_apparatus(self) -> List[str]:
        """
        获取尚未检测到的必要器材列表

        Returns:
            缺失器材名称列表
        """
        completeness = self.check_apparatus_completeness()
        return [name for name, present in completeness.items() if not present]

    # ── 统计与重置 ────────────────────────────────────────────────────────────

    def get_statistics(self) -> Dict:
        """获取时序分析统计信息"""
        return {
            "tracked_objects": len(self.object_tracks),
            "history_length": len(self.detection_history),
            "identified_actions": len(self.identified_actions),
            "object_names": list(self.object_tracks.keys()),
            "current_step": self.current_step,
            "step_elapsed_time": self.get_step_elapsed_time(),
            "safety_events_count": len(self.safety_events),
            "missing_apparatus": self.get_missing_apparatus(),
        }

    def reset(self) -> None:
        """重置分析器状态"""
        self.detection_history.clear()
        self.object_tracks.clear()
        self.identified_actions.clear()
        self.safety_events.clear()
        self.current_step = "准备器材"
        self.step_start_time = time.time()
        logger.info("时序分析器已重置")
