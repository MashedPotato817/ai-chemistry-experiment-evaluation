"""
评分引擎实现

实现熟练度评分算法：
- S1 时长得分：基于正态分布，公式 f(t) = 2 - 2Φ((μ-t)/σ)
- S2 表现得分：基于操作质量、错误次数、安全合规等综合评分
- 最终得分：S = 0.7 × S1 + 0.3 × S2

针对气体制备与收集实验（fs_s_best.pt 模型）优化默认参数。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from scipy.stats import norm

from ..utils.logger import get_logger
from ..database.dao import ExperimentStatisticsDAO

logger = get_logger(__name__)

# ── 实验默认统计参数 ──────────────────────────────────────────────────────────
# 基于气体制备与收集实验的经验估计值
# 随着数据积累会自动更新
DEFAULT_EXPERIMENT_STATS: Dict[str, Dict[str, float]] = {
    "气体制备与收集实验": {
        "mu": 480.0,    # 8分钟（含装置组装和气密性检验）
        "sigma": 90.0,  # 1.5分钟标准差
    },
    "双氧水制氧气": {
        "mu": 300.0,
        "sigma": 60.0,
    },
}

# 安全违规的额外惩罚系数
SAFETY_VIOLATION_PENALTY = 0.6  # 违规时 S2 最高只能得 0.6 × 2 = 1.2 分


@dataclass
class ScoreResult:
    """评分结果"""
    s1_time_score: float            # S1 时长得分 [0, 2]
    s2_performance_score: float     # S2 表现得分 [0, 2]
    final_score: float              # 最终得分 [0, 2]
    percentile_rank: float          # 百分位排名 [0, 100]
    calculation_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "s1_time_score": self.s1_time_score,
            "s2_performance_score": self.s2_performance_score,
            "final_score": self.final_score,
            "percentile_rank": self.percentile_rank,
            "calculation_details": self.calculation_details,
        }

    def get_grade(self) -> str:
        """根据最终得分返回等级"""
        if self.final_score >= 1.8:
            return "优秀"
        elif self.final_score >= 1.4:
            return "良好"
        elif self.final_score >= 1.0:
            return "合格"
        else:
            return "需改进"

    def get_summary(self) -> str:
        """返回得分摘要文本"""
        return (
            f"最终得分: {self.final_score:.2f}/2.00 ({self.get_grade()}) | "
            f"时长得分: {self.s1_time_score:.2f} | "
            f"表现得分: {self.s2_performance_score:.2f} | "
            f"超越 {self.percentile_rank:.1f}% 的学生"
        )


class ScoringEngine:
    """
    评分引擎

    负责计算实验熟练度评分，支持气体制备与收集实验。
    """

    S1_WEIGHT = 0.7  # 时长得分权重
    S2_WEIGHT = 0.3  # 表现得分权重

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化评分引擎

        Args:
            db_path: 数据库路径（用于获取/更新统计参数）
        """
        self.db_path = db_path
        self.stats_dao = ExperimentStatisticsDAO(db_path) if db_path else None
        logger.info(
            f"评分引擎初始化 - S1权重: {self.S1_WEIGHT}, S2权重: {self.S2_WEIGHT}"
        )

    # ── S1 时长得分 ───────────────────────────────────────────────────────────

    def calculate_s1_time_score(
        self,
        actual_time: float,
        mu: float,
        sigma: float,
    ) -> float:
        """
        计算 S1 时长得分

        公式: f(t) = 2 - 2Φ((μ-t)/σ)
        - 完成时间越接近均值，得分越高（接近 1.0）
        - 完成时间远快于均值，得分接近 2.0
        - 完成时间远慢于均值，得分接近 0.0

        Args:
            actual_time: 实际完成时间（秒）
            mu: 历史平均完成时间（秒）
            sigma: 历史标准差（秒）

        Returns:
            S1 得分 [0, 2]
        """
        if sigma <= 0:
            logger.warning(f"标准差无效: {sigma}，使用默认值 60.0")
            sigma = 60.0

        z_score = (mu - actual_time) / sigma
        phi_z = norm.cdf(z_score)
        # 完成越快（actual_time < mu）→ z > 0 → Φ > 0.5 → S1 > 1.0
        # 完成越慢（actual_time > mu）→ z < 0 → Φ < 0.5 → S1 < 1.0
        s1_score = max(0.0, min(2.0, 2.0 * phi_z))

        logger.debug(
            f"S1: actual={actual_time:.0f}s, μ={mu:.0f}s, σ={sigma:.0f}s, "
            f"z={z_score:.3f}, Φ={phi_z:.3f}, S1={s1_score:.3f}"
        )
        return s1_score

    # ── S2 表现得分 ───────────────────────────────────────────────────────────

    def calculate_s2_performance_score(
        self,
        error_count: int,
        question_count: int,
        operation_quality: float,
        safety_compliance: bool = True,
        safety_event_count: int = 0,
        steps_completed: int = 0,
        total_steps: int = 9,
    ) -> float:
        """
        计算 S2 表现得分

        综合考虑：
        - 步骤完成率（作为乘数，未完成步骤直接压低得分）
        - 操作质量（LLM 评估，0-1）
        - 操作错误次数
        - 提问次数（适度提问不扣分）
        - 安全规范遵守（违规有额外惩罚）
        - 安全事件次数

        Args:
            error_count: 操作错误次数
            question_count: 提问次数
            operation_quality: 操作质量评分（0-1）
            safety_compliance: 是否遵守安全规范
            safety_event_count: 安全事件次数（手距危险器材过近等）
            steps_completed: 完成的步骤数
            total_steps: 总步骤数

        Returns:
            S2 得分 [0, 2]
        """
        # 步骤完成率：作为整体乘数，完成率越低得分越低
        # 完成率 = steps_completed / total_steps，范围 [0, 1]
        # 乘数公式：0.4 + 0.6 × completion_rate
        #   - 全部完成（100%）→ 乘数 1.0，不影响得分
        #   - 完成一半（50%） → 乘数 0.7，得分打七折
        #   - 未完成任何步骤  → 乘数 0.4，得分打四折
        if total_steps > 0:
            completion_rate = min(steps_completed / total_steps, 1.0)
        else:
            completion_rate = 1.0
        completion_multiplier = 0.4 + 0.6 * completion_rate

        # 基础分：操作质量映射到 [0, 2]
        base_score = operation_quality * 2.0

        # 错误惩罚：每个错误扣 0.1 分，最多扣 1.0 分
        error_penalty = min(error_count * 0.1, 1.0)

        # 提问惩罚：0-2次不扣，3-5次每次扣0.05，6次以上每次扣0.1
        if question_count <= 2:
            question_penalty = 0.0
        elif question_count <= 5:
            question_penalty = (question_count - 2) * 0.05
        else:
            question_penalty = 3 * 0.05 + (question_count - 5) * 0.1
        question_penalty = min(question_penalty, 0.5)

        # 安全事件惩罚：每次安全事件扣 0.1 分，最多扣 0.5 分
        safety_event_penalty = min(safety_event_count * 0.1, 0.5)

        # 计算扣罚后的基础分，再乘以步骤完成率乘数
        raw_score = base_score - error_penalty - question_penalty - safety_event_penalty
        s2_score = raw_score * completion_multiplier

        # 安全规范违规：上限降低到 SAFETY_VIOLATION_PENALTY × 2
        if not safety_compliance:
            s2_score = min(s2_score, 2.0 * SAFETY_VIOLATION_PENALTY)

        s2_score = max(0.0, min(2.0, s2_score))

        logger.debug(
            f"S2: base={base_score:.3f}, completion={completion_rate:.2f}×"
            f"(multiplier={completion_multiplier:.2f}), "
            f"err_pen={error_penalty:.3f}, q_pen={question_penalty:.3f}, "
            f"safe_pen={safety_event_penalty:.3f}, S2={s2_score:.3f}"
        )
        return s2_score

    # ── 最终得分 ──────────────────────────────────────────────────────────────

    def compute_final_score(self, s1: float, s2: float) -> float:
        """
        计算最终熟练度评分

        公式: S = 0.7 × S1 + 0.3 × S2

        Args:
            s1: S1 时长得分
            s2: S2 表现得分

        Returns:
            最终得分 [0, 2]
        """
        final_score = max(0.0, min(2.0, self.S1_WEIGHT * s1 + self.S2_WEIGHT * s2))
        logger.info(
            f"最终得分: S1={s1:.3f}×{self.S1_WEIGHT} + "
            f"S2={s2:.3f}×{self.S2_WEIGHT} = {final_score:.3f}"
        )
        return final_score

    def calculate_percentile_rank(
        self,
        final_score: float,
        mu_score: float = 1.0,
        sigma_score: float = 0.35,
    ) -> float:
        """
        基于最终得分计算百分位排名（超越了多少百分比的学生）

        使用最终得分的正态分布估计，得分越高超越比例越高。
        默认参数基于得分范围 [0,2]、均值约1.0、标准差约0.35的经验估计。

        Args:
            final_score: 最终得分 [0, 2]
            mu_score: 得分分布均值（默认 1.0）
            sigma_score: 得分分布标准差（默认 0.35）

        Returns:
            超越百分比 [0, 100]，即"超越了 X% 的学生"
        """
        if sigma_score <= 0:
            sigma_score = 0.35
        z_score = (final_score - mu_score) / sigma_score
        percentile = norm.cdf(z_score) * 100.0
        return max(0.0, min(100.0, percentile))

    # ── 统计参数管理 ──────────────────────────────────────────────────────────

    def get_experiment_statistics(self, experiment_type: str) -> Tuple[float, float]:
        """
        获取实验类型的统计参数（μ, σ）

        优先从数据库读取，其次使用内置默认值。

        Args:
            experiment_type: 实验类型名称

        Returns:
            (平均时长, 标准差) 元组
        """
        if self.stats_dao:
            try:
                stats = self.stats_dao.get_statistics(experiment_type)
                if stats and stats["sample_count"] >= 5:
                    # 样本数 >= 5 才使用数据库统计，避免小样本偏差
                    return stats["mean_duration"], stats["std_deviation"]
            except Exception as e:
                logger.warning(f"获取统计参数失败: {e}")

        # 使用内置默认值
        defaults = DEFAULT_EXPERIMENT_STATS.get(
            experiment_type,
            DEFAULT_EXPERIMENT_STATS["气体制备与收集实验"],
        )
        mu, sigma = defaults["mu"], defaults["sigma"]
        logger.info(f"使用默认统计参数 [{experiment_type}]: μ={mu}s, σ={sigma}s")
        return mu, sigma

    # ── 完整评分计算 ──────────────────────────────────────────────────────────

    def calculate_complete_score(
        self,
        experiment_type: str,
        actual_time: float,
        error_count: int,
        question_count: int,
        operation_quality: float,
        safety_compliance: bool = True,
        safety_event_count: int = 0,
        steps_completed: int = 0,
        total_steps: int = 9,
        custom_mu: Optional[float] = None,
        custom_sigma: Optional[float] = None,
    ) -> ScoreResult:
        """
        计算完整的评分结果

        Args:
            experiment_type: 实验类型
            actual_time: 实际完成时间（秒）
            error_count: 错误次数
            question_count: 提问次数
            operation_quality: 操作质量（0-1）
            safety_compliance: 安全规范遵守
            safety_event_count: 安全事件次数
            steps_completed: 完成的步骤数
            total_steps: 总步骤数
            custom_mu: 自定义平均时长（可选，覆盖数据库值）
            custom_sigma: 自定义标准差（可选，覆盖数据库值）

        Returns:
            ScoreResult 对象
        """
        # 获取统计参数
        if custom_mu is not None and custom_sigma is not None:
            mu, sigma = custom_mu, custom_sigma
        else:
            mu, sigma = self.get_experiment_statistics(experiment_type)

        # 计算各项得分
        s1_score = self.calculate_s1_time_score(actual_time, mu, sigma)
        s2_score = self.calculate_s2_performance_score(
            error_count=error_count,
            question_count=question_count,
            operation_quality=operation_quality,
            safety_compliance=safety_compliance,
            safety_event_count=safety_event_count,
            steps_completed=steps_completed,
            total_steps=total_steps,
        )
        final_score = self.compute_final_score(s1_score, s2_score)
        percentile = self.calculate_percentile_rank(final_score)

        result = ScoreResult(
            s1_time_score=s1_score,
            s2_performance_score=s2_score,
            final_score=final_score,
            percentile_rank=percentile,
            calculation_details={
                "experiment_type": experiment_type,
                "actual_time": actual_time,
                "mu": mu,
                "sigma": sigma,
                "error_count": error_count,
                "question_count": question_count,
                "operation_quality": operation_quality,
                "safety_compliance": safety_compliance,
                "safety_event_count": safety_event_count,
                "steps_completed": steps_completed,
                "total_steps": total_steps,
                "s1_weight": self.S1_WEIGHT,
                "s2_weight": self.S2_WEIGHT,
            },
        )

        logger.info(
            f"评分完成 [{experiment_type}]: "
            f"S1={s1_score:.3f}, S2={s2_score:.3f}, "
            f"Final={final_score:.3f} ({result.get_grade()}), "
            f"百分位={percentile:.1f}%"
        )
        return result

    # ── 统计参数更新 ──────────────────────────────────────────────────────────

    def update_experiment_statistics(
        self,
        experiment_type: str,
        new_duration: float,
    ) -> None:
        """
        用 Welford 在线算法增量更新实验统计参数

        Args:
            experiment_type: 实验类型
            new_duration: 新的实验时长（秒）
        """
        if not self.stats_dao:
            logger.warning("未配置数据库，无法更新统计参数")
            return

        try:
            stats = self.stats_dao.get_statistics(experiment_type)

            if stats:
                n = stats["sample_count"]
                old_mu = stats["mean_duration"]
                old_sigma = stats["std_deviation"]

                new_mu = (old_mu * n + new_duration) / (n + 1)
                new_variance = (
                    old_sigma ** 2 * n
                    + (new_duration - old_mu) * (new_duration - new_mu)
                ) / (n + 1)
                new_sigma = max(1.0, np.sqrt(new_variance))  # 最小标准差 1 秒

                self.stats_dao.update_statistics(
                    experiment_type=experiment_type,
                    mean_duration=new_mu,
                    std_deviation=new_sigma,
                    sample_count=n + 1,
                )
                logger.info(
                    f"统计参数更新 [{experiment_type}]: "
                    f"μ={new_mu:.1f}s, σ={new_sigma:.1f}s, n={n+1}"
                )
            else:
                # 首次记录，使用默认标准差
                defaults = DEFAULT_EXPERIMENT_STATS.get(
                    experiment_type,
                    DEFAULT_EXPERIMENT_STATS["气体制备与收集实验"],
                )
                self.stats_dao.create_statistics(
                    experiment_type=experiment_type,
                    mean_duration=new_duration,
                    std_deviation=defaults["sigma"],
                    sample_count=1,
                )
                logger.info(f"创建统计记录 [{experiment_type}]: 首次时长={new_duration:.1f}s")

        except Exception as e:
            logger.error(f"更新统计参数失败: {e}")
