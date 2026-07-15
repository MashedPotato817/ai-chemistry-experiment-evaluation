"""
大模型交互服务实现

提供实时交互、智能引导和性能分析功能。
针对气体制备与收集实验（fs_s_best.pt 模型）优化。
"""

import hashlib
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI

from ..config.config import config as app_config
from ..utils.logger import get_logger
from ..utils.exceptions import LLMError
from ..detection.models import Action, ExperimentStep

logger = get_logger(__name__)

# 实验名称（与模型对应）
EXPERIMENT_NAME = "气体制备与收集实验"

# 实验步骤序列（与 temporal_analyzer 保持一致）
EXPERIMENT_STEPS = [
    "准备器材", "添加药品", "组装装置", "检查气密性",
    "点燃酒精灯", "收集气体", "验证气体", "熄灭酒精灯", "整理器材",
]

# 各步骤的降级指导（无 API 时使用）
FALLBACK_GUIDANCE: Dict[str, str] = {
    "准备器材":   "请检查锥形瓶、双孔橡胶塞、导管、集气瓶等器材是否齐全，并清洗干净。",
    "添加药品":   "用药勺取适量二氧化锰固体放入锥形瓶，注意不要撒落。",
    "组装装置":   "将双孔橡胶塞插入锥形瓶，连接导管和橡胶管，夹好止水夹。",
    "检查气密性": "关闭止水夹，将导管末端插入水中，用手捂住锥形瓶，观察是否有气泡冒出。",
    "点燃酒精灯": "用火柴点燃酒精灯，注意不要用酒精灯引燃另一个酒精灯。",
    "收集气体":   "将导管伸入集气瓶底部，待气泡连续均匀冒出后开始收集。",
    "验证气体":   "将带火星的木条伸入集气瓶，若木条复燃则证明收集到氧气。",
    "熄灭酒精灯": "用灯帽盖灭酒精灯，不可用嘴吹灭。",
    "整理器材":   "将废液倒入废液缸，用洗瓶清洗器材，整理归位。",
}


@dataclass
class ExperimentContext:
    """实验上下文"""
    current_step: str           # 当前步骤
    detected_objects: List[str] # 当前帧检测到的器材列表
    detected_actions: List[Action]  # 检测到的动作
    user_questions: List[str]   # 用户提问
    error_count: int            # 错误次数
    elapsed_time: float         # 已用时间（秒）
    safety_events: List[Dict]   # 安全事件列表
    missing_apparatus: List[str] = None  # 缺失器材


@dataclass
class PerformanceAnalysis:
    """性能分析结果"""
    error_types: List[str]          # 错误类型列表
    question_frequency: float       # 提问频率（次/分钟）
    operation_quality_score: float  # 操作质量得分（0-1）
    safety_compliance: bool         # 安全规范遵守
    detailed_feedback: str          # 详细反馈


class LLMInteractionService:
    """
    大模型交互服务类

    负责与大模型 API 交互，提供智能引导和分析。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """
        初始化大模型交互服务

        Args:
            api_key: API 密钥（优先级高于 .env 配置）
            model_name: 模型名称（优先级高于 .env 配置）
            base_url: API 基础 URL（优先级高于 .env 配置）
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
        """
        llm_cfg = app_config.llm

        # 参数优先级：显式传入 > .env 配置
        resolved_api_key = api_key or llm_cfg.api_key
        self.model_name = model_name or llm_cfg.model_name
        resolved_base_url = base_url or llm_cfg.api_base
        self.timeout = timeout if timeout is not None else float(llm_cfg.timeout)
        self.max_retries = max_retries if max_retries is not None else llm_cfg.max_retries

        # 初始化 OpenAI 兼容客户端（支持 Kimi / Qwen / DeepSeek 等）
        client_kwargs: Dict[str, Any] = {"base_url": resolved_base_url}
        if resolved_api_key:
            client_kwargs["api_key"] = resolved_api_key
        self.client = OpenAI(**client_kwargs)

        # 响应缓存 {cache_key: response_text}
        self.response_cache: Dict[str, str] = {}

        logger.info(
            f"大模型服务初始化 - 模型: {self.model_name}, "
            f"API地址: {resolved_base_url}, "
            f"超时: {self.timeout}s, 重试: {self.max_retries}次"
        )

    # ── API 调用 ──────────────────────────────────────────────────────────────

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> str:
        """
        调用大模型 API（带重试）

        Args:
            messages: 消息列表
            temperature: 温度参数

        Returns:
            模型响应文本

        Raises:
            LLMError: API 调用失败
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    timeout=self.timeout,
                )
                return response.choices[0].message.content

            except openai.APITimeoutError as e:
                logger.warning(f"API 超时 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise LLMError(f"API 超时，已重试 {self.max_retries} 次")
                time.sleep(2 ** attempt)  # 指数退避

            except openai.RateLimitError as e:
                logger.warning(f"API 速率限制: {e}")
                if attempt == self.max_retries - 1:
                    raise LLMError("API 速率限制，请稍后重试")
                time.sleep(5)

            except openai.APIError as e:
                logger.error(f"API 错误: {e}")
                raise LLMError(f"API 调用失败: {e}")

            except Exception as e:
                logger.error(f"未知错误: {e}")
                raise LLMError(f"调用失败: {e}")

    def _make_cache_key(self, *args) -> str:
        """生成稳定的缓存键（包含所有参数的哈希）"""
        content = "|".join(str(a) for a in args)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    # ── 实时指导 ──────────────────────────────────────────────────────────────

    def provide_guidance(self, context: ExperimentContext) -> str:
        """
        提供实时实验指导

        Args:
            context: 实验上下文

        Returns:
            指导文本
        """
        try:
            # 缓存键包含会影响回答的上下文，避免将旧问题的回答复用给新问题。
            objects_key = ",".join(sorted(context.detected_objects))
            latest_question = context.user_questions[-1] if context.user_questions else ""
            cache_key = self._make_cache_key(
                context.current_step, context.error_count, objects_key, latest_question
            )

            if cache_key in self.response_cache:
                logger.debug("使用缓存的指导响应")
                return self.response_cache[cache_key]

            prompt = self._build_guidance_prompt(context)
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"你是一位经验丰富的化学实验指导老师，"
                        f"正在指导学生进行{EXPERIMENT_NAME}。"
                        f"实验步骤依次为：{'→'.join(EXPERIMENT_STEPS)}。"
                        f"请根据学生当前的操作状态给出简洁、具体的指导。"
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            response = self._call_api(messages, temperature=0.7)
            self.response_cache[cache_key] = response

            logger.info(f"提供实验指导 - 步骤: {context.current_step}")
            return response

        except LLMError as e:
            logger.error(f"提供指导失败: {e}")
            return self._get_fallback_guidance(context)
        except Exception as e:
            logger.error(f"未知错误: {e}")
            return self._get_fallback_guidance(context)

    def _build_guidance_prompt(self, context: ExperimentContext) -> str:
        """构建指导提示"""
        # 安全警告部分
        safety_warning = ""
        if context.safety_events:
            recent = context.safety_events[-1]
            safety_warning = f"\n⚠️ 安全警告: {recent.get('message', '')}"

        # 缺失器材提示
        missing_hint = ""
        if context.missing_apparatus:
            missing_hint = f"\n尚未检测到的器材: {', '.join(context.missing_apparatus)}"

        question_hint = ""
        if context.user_questions:
            question_hint = f"\n学生刚刚的问题: {context.user_questions[-1]}"

        prompt = f"""
实验名称: {EXPERIMENT_NAME}
当前步骤: {context.current_step}
已用时间: {context.elapsed_time:.0f}秒
错误次数: {context.error_count}
当前画面中的器材: {', '.join(context.detected_objects) if context.detected_objects else '无'}
最近动作: {self._format_actions(context.detected_actions[-3:])}
{safety_warning}{missing_hint}{question_hint}

请用不超过80字给出当前步骤的具体操作指导；若学生提问，先直接回答问题：
""".strip()
        return prompt

    # ── 表现分析 ──────────────────────────────────────────────────────────────

    def analyze_performance(
        self,
        experiment_steps: List[ExperimentStep],
        user_questions: List[str],
        detected_errors: List[str],
        total_time: float,
        safety_events: Optional[List[Dict]] = None,
    ) -> PerformanceAnalysis:
        """
        分析用户实验表现

        Args:
            experiment_steps: 实验步骤列表
            user_questions: 用户提问列表
            detected_errors: 检测到的错误列表
            total_time: 总时间（秒）
            safety_events: 安全事件列表

        Returns:
            PerformanceAnalysis 对象
        """
        try:
            prompt = self._build_analysis_prompt(
                experiment_steps, user_questions, detected_errors,
                total_time, safety_events or []
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"你是一位化学实验评估专家，负责分析学生进行{EXPERIMENT_NAME}的表现。"
                        f"请客观、专业地评估操作质量，重点关注安全规范和操作准确性。"
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            response = self._call_api(messages, temperature=0.3)
            analysis = self._parse_performance_analysis(
                response, user_questions, detected_errors, total_time
            )

            logger.info(f"性能分析完成 - 质量得分: {analysis.operation_quality_score:.2f}")
            return analysis

        except Exception as e:
            logger.error(f"性能分析失败: {e}")
            return self._get_fallback_analysis(user_questions, detected_errors)

    def _build_analysis_prompt(
        self,
        steps: List[ExperimentStep],
        questions: List[str],
        errors: List[str],
        total_time: float,
        safety_events: List[Dict],
    ) -> str:
        """构建分析提示"""
        completed_steps = len(steps)
        total_steps = len(EXPERIMENT_STEPS)
        completion_rate = completed_steps / total_steps * 100 if total_steps > 0 else 0

        safety_summary = ""
        if safety_events:
            safety_summary = f"\n安全事件 ({len(safety_events)}次):\n" + "\n".join(
                f"  - {e.get('message', '')}" for e in safety_events[:5]
            )

        prompt = f"""
请分析以下{EXPERIMENT_NAME}的实验表现：

完成步骤: {completed_steps}/{total_steps}（完成率 {completion_rate:.0f}%）
总用时: {total_time:.0f}秒（{total_time/60:.1f}分钟）
提问次数: {len(questions)}
操作错误次数: {len(errors)}
{safety_summary}

检测到的错误:
{chr(10).join(f"  - {e}" for e in errors) if errors else "  无"}

请按以下格式评估（严格遵守格式）：
质量得分: [0到1之间的小数，精确到两位]
安全规范: [是/否]
错误类型: [用中文逗号分隔，无则填"无"]
反馈: [不超过150字的详细反馈]
""".strip()
        return prompt

    def _parse_performance_analysis(
        self,
        response: str,
        questions: List[str],
        errors: List[str],
        total_time: float,
    ) -> PerformanceAnalysis:
        """解析性能分析响应"""
        quality_score = 0.75
        safety_compliance = True
        error_types: List[str] = []
        feedback = response

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("质量得分"):
                try:
                    quality_score = float(line.split(":")[-1].strip())
                    quality_score = max(0.0, min(1.0, quality_score))
                except ValueError:
                    pass
            elif line.startswith("安全规范"):
                safety_compliance = "是" in line
            elif line.startswith("错误类型"):
                raw = line.split(":", 1)[-1].strip()
                error_types = [t.strip() for t in raw.replace("，", ",").split(",") if t.strip() and t.strip() != "无"]
            elif line.startswith("反馈"):
                feedback = line.split(":", 1)[-1].strip()

        question_frequency = len(questions) / (total_time / 60.0) if total_time > 0 else 0.0

        return PerformanceAnalysis(
            error_types=error_types or [e[:50] for e in errors],
            question_frequency=question_frequency,
            operation_quality_score=quality_score,
            safety_compliance=safety_compliance,
            detailed_feedback=feedback,
        )

    # ── 推荐生成 ──────────────────────────────────────────────────────────────

    def generate_recommendations(self, analysis: PerformanceAnalysis) -> List[str]:
        """
        生成个性化改进推荐

        Args:
            analysis: 性能分析结果

        Returns:
            推荐列表（最多5条）
        """
        try:
            prompt = f"""
基于以下{EXPERIMENT_NAME}的表现，推荐3-5个具体的改进建议：

操作质量: {analysis.operation_quality_score:.2f}（满分1.0）
安全规范: {"遵守" if analysis.safety_compliance else "未遵守"}
主要错误: {', '.join(analysis.error_types) if analysis.error_types else "无"}
提问频率: {analysis.question_frequency:.1f}次/分钟

请给出针对性的练习建议，每条建议一行，不加序号：
""".strip()

            messages = [
                {
                    "role": "system",
                    "content": f"你是化学实验教学专家，根据学生在{EXPERIMENT_NAME}中的表现推荐针对性练习。",
                },
                {"role": "user", "content": prompt},
            ]

            response = self._call_api(messages, temperature=0.7)
            recommendations = [
                line.strip().lstrip("0123456789.-）) ")
                for line in response.split("\n")
                if line.strip() and len(line.strip()) > 5
            ]

            logger.info(f"生成 {len(recommendations)} 条推荐")
            return recommendations[:5]

        except Exception as e:
            logger.error(f"生成推荐失败: {e}")
            return self._get_fallback_recommendations(analysis)

    # ── 降级处理 ──────────────────────────────────────────────────────────────

    def _get_fallback_guidance(self, context: ExperimentContext) -> str:
        """无 API 时的降级指导（基于当前步骤）"""
        # 优先返回当前步骤的指导
        if context.current_step in FALLBACK_GUIDANCE:
            guidance = FALLBACK_GUIDANCE[context.current_step]
        else:
            guidance = "请按照实验步骤继续操作，注意安全。"

        # 如果有安全事件，附加警告
        if context.safety_events:
            recent = context.safety_events[-1]
            guidance += f" ⚠️ {recent.get('message', '')}"

        return guidance

    def _get_fallback_analysis(
        self,
        questions: List[str],
        errors: List[str],
    ) -> PerformanceAnalysis:
        """无 API 时的降级分析"""
        quality_score = max(0.4, 1.0 - len(errors) * 0.1)
        safety_compliance = len(errors) < 3

        return PerformanceAnalysis(
            error_types=[e[:50] for e in errors],
            question_frequency=len(questions) / 5.0,
            operation_quality_score=quality_score,
            safety_compliance=safety_compliance,
            detailed_feedback="实验完成。建议多加练习，熟悉气体制备与收集的标准操作流程。",
        )

    def _get_fallback_recommendations(self, analysis: PerformanceAnalysis) -> List[str]:
        """无 API 时的降级推荐"""
        recommendations = [
            "复习气体制备与收集实验的标准操作规范",
            "练习装置气密性检验的正确方法",
            "熟悉酒精灯的安全使用规范",
        ]
        if analysis.operation_quality_score < 0.6:
            recommendations.append("重复练习本实验，提高操作熟练度")
        if not analysis.safety_compliance:
            recommendations.append("重点学习实验安全规范，特别是明火操作注意事项")
        if analysis.question_frequency > 2.0:
            recommendations.append("实验前充分预习实验原理和步骤")
        return recommendations[:5]

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def _format_actions(self, actions: List[Action]) -> str:
        """格式化动作列表"""
        if not actions:
            return "无"
        return "、".join(
            f"{a.action_type}({a.duration:.1f}s)" for a in actions
        )

    def clear_cache(self) -> None:
        """清空响应缓存"""
        self.response_cache.clear()
        logger.info("响应缓存已清空")
