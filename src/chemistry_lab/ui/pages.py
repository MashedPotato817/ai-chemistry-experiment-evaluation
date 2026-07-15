"""
Streamlit 页面组件

针对气体制备与收集实验（fs_s_best.pt 模型）优化的 UI。
"""

import time
from datetime import datetime
from typing import Optional, List

import streamlit as st

from ..user_management.user_manager import UserManager
from ..database.dao import ExperimentDAO
from ..detection.models import ExperimentStep
from ..llm.llm_service import ExperimentContext, LLMInteractionService, PerformanceAnalysis
from ..utils.logger import get_logger

logger = get_logger(__name__)

# 实验类型（与模型对应）
# 注意：目前只有"气体制备与收集实验（双氧水制氧气）"可以真正运行，其他为展示用
EXPERIMENT_TYPES = [
    "气体制备与收集实验（双氧水制氧气）",  # ✅ 可运行
    "气体制备与收集实验（高锰酸钾制氧气）",  # 展示用
    "酸碱中和滴定实验",  # 展示用
    "金属活动性顺序探究实验",  # 展示用
    "粗盐提纯实验",  # 展示用
    "二氧化碳制取与性质实验",  # 展示用
    "燃烧条件探究实验",  # 展示用
    "溶液配制实验",  # 展示用
    "酸碱盐性质实验",  # 展示用
    "金属与酸反应实验",  # 展示用
    "氢气还原氧化铜实验",  # 展示用
    "碳酸钠与盐酸反应实验",  # 展示用
    "铁生锈条件探究实验",  # 展示用
]

# 可实际运行的实验类型
AVAILABLE_EXPERIMENTS = ["气体制备与收集实验（双氧水制氧气）"]

# 实验步骤（与 temporal_analyzer 保持一致）
EXPERIMENT_STEPS = [
    "准备器材", "添加药品", "组装装置", "检查气密性",
    "点燃酒精灯", "收集气体", "验证气体", "熄灭酒精灯", "整理器材",
]

# 实验所需器材清单
REQUIRED_APPARATUS = [
    "锥形瓶", "双孔橡胶塞", "直角导管", "橡胶管",
    "集气瓶", "玻璃片", "止水夹", "长颈漏斗",
    "酒精灯", "火柴", "药勺", "二氧化锰固体",
    "废液缸", "洗瓶",
]


# ── 登录页面 ──────────────────────────────────────────────────────────────────

def show_login_page() -> None:
    """显示登录页面"""
    st.subheader("🔐 用户登录")

    with st.form("login_form"):
        username = st.text_input("用户名", placeholder="请输入用户名")
        password = st.text_input("密码", type="password", placeholder="请输入密码")
        remember_me = st.checkbox("记住密码")
        submitted = st.form_submit_button("登录", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("请输入用户名和密码")
                return
            try:
                from ..user_management.models import LoginCredentials
                user_manager = UserManager()
                result = user_manager.authenticate_user(
                    LoginCredentials(username=username, password=password)
                )
                if result.success and result.user_profile:
                    st.session_state.logged_in = True
                    st.session_state.user_id = result.user_profile.user_id
                    st.session_state.username = result.user_profile.username
                    st.success("✅ 登录成功！")
                    logger.info(f"用户登录成功: {username}")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"❌ {result.error_message or '用户名或密码错误'}")
            except Exception as e:
                logger.error(f"登录异常: {e}")
                st.error(f"❌ 登录失败: {e}")


# ── 注册页面 ──────────────────────────────────────────────────────────────────

def show_register_page() -> None:
    """显示注册页面"""
    st.subheader("📝 用户注册")

    with st.form("register_form"):
        username = st.text_input("用户名", placeholder="3-20个字符")
        password = st.text_input("密码", type="password", placeholder="至少6个字符")
        password_confirm = st.text_input("确认密码", type="password", placeholder="再次输入密码")
        email = st.text_input("邮箱（可选）", placeholder="your@email.com")
        submitted = st.form_submit_button("注册", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("用户名和密码不能为空")
                return
            if len(username) < 3 or len(username) > 20:
                st.error("用户名长度应为 3-20 个字符")
                return
            if not password:
                st.error("密码不能为空")
                return
            if password != password_confirm:
                st.error("两次输入的密码不一致")
                return
            try:
                from ..user_management.models import RegistrationData
                user_manager = UserManager()
                result = user_manager.register_user(
                    RegistrationData(
                        username=username,
                        password=password,
                        email=email or None,
                        confirm_password=password_confirm,
                    )
                )
                if result.success:
                    st.success("✅ 注册成功！请切换到登录页面")
                    logger.info(f"用户注册成功: {username}")
                else:
                    st.error(f"❌ 注册失败: {result.error_message}")
            except Exception as e:
                logger.error(f"注册异常: {e}")
                st.error(f"❌ 注册失败: {e}")


# ── 主页 ──────────────────────────────────────────────────────────────────────

def show_main_page() -> None:
    """显示主页"""
    st.header("🏠 主页")

    # 从数据库读取用户统计
    user_id = st.session_state.get("user_id")
    total_exp, avg_score, best_score = _get_user_stats(user_id)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总实验次数", total_exp)
    with col2:
        st.metric("平均得分", f"{avg_score:.2f}" if avg_score else "—")
    with col3:
        st.metric("最高得分", f"{best_score:.2f}" if best_score else "—")

    st.markdown("---")
    st.subheader("📊 系统功能")

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "### 🔬 实时检测\n"
            "- YOLOv11s 模型识别 19 种实验器材\n"
            "- 时序行为分析与步骤推断\n"
            "- 安全事件实时预警"
        )
    with col2:
        st.info(
            "### 🤖 智能交互\n"
            "- 大模型实时操作指导\n"
            "- 个性化改进建议\n"
            "- S1/S2 双维度评分报告"
        )

    st.markdown("---")
    st.subheader("🧪 实验器材清单")
    cols = st.columns(4)
    for i, apparatus in enumerate(REQUIRED_APPARATUS):
        cols[i % 4].write(f"• {apparatus}")

    st.markdown("---")
    st.subheader("🎯 快速开始")
    st.write("点击侧边栏的「开始实验」开始你的气体制备与收集实验评估！")


def _get_user_stats(user_id: Optional[int]):
    """从数据库获取用户统计数据"""
    if not user_id:
        return 0, None, None
    try:
        dao = ExperimentDAO()
        experiments, total = dao.get_user_experiments(user_id, page_size=1000)
        if not experiments:
            return 0, None, None
        completed = [e for e in experiments if e.final_score is not None]
        if not completed:
            return total, None, None
        scores = [e.final_score for e in completed]
        return total, sum(scores) / len(scores), max(scores)
    except Exception as e:
        logger.error(f"获取用户统计失败: {e}")
        return 0, None, None


# ── 实验页面 ──────────────────────────────────────────────────────────────────

def show_experiment_page() -> None:
    """显示实验页面"""
    st.header("🔬 开始实验")

    # 实验类型选择
    experiment_type = st.selectbox(
        "选择实验类型", 
        EXPERIMENT_TYPES,
        help="目前仅「气体制备与收集实验」可运行演示"
    )
    
    # 检查是否为可运行的实验类型
    if experiment_type not in AVAILABLE_EXPERIMENTS:
        st.warning(f"⚠️ 「{experiment_type}」功能正在开发中，敬请期待！")
        st.info("💡 请选择「气体制备与收集实验（双氧水制氧气）」进行演示")
        return

    st.markdown("---")

    # 初始化会话状态
    for key, default in [
        ("experiment_running", False),
        ("experiment_finished", False),
        ("experiment_result", None),
        ("experiment_start_time", None),
        ("current_step_index", 0),
        ("detected_objects", []),
        ("error_count", 0),
        ("question_count", 0),
        ("user_questions", []),
        ("safety_events", []),
        ("actions_log", []),
        ("completed_steps", []),
        ("step_started_at", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if st.session_state.experiment_finished:
        _show_experiment_result(experiment_type)
    elif not st.session_state.experiment_running:
        _show_pre_experiment(experiment_type)
    else:
        _show_running_experiment(experiment_type)


def _show_pre_experiment(experiment_type: str) -> None:
    """实验开始前的准备界面"""
    st.info(f"准备开始「{experiment_type}」")

    # 实验步骤预览
    with st.expander("📋 查看实验步骤", expanded=True):
        for i, step in enumerate(EXPERIMENT_STEPS, 1):
            st.write(f"{i}. {step}")

    # 器材检查提示
    with st.expander("🔧 器材准备清单"):
        # 初始化全选状态
        if 'apparatus_all_checked' not in st.session_state:
            st.session_state.apparatus_all_checked = False

        # 一键全部勾选按钮
        btn_col, status_col = st.columns([1, 3])
        with btn_col:
            if st.button("✅ 一键全部勾选", key="check_all_apparatus"):
                st.session_state.apparatus_all_checked = True
                for apparatus in REQUIRED_APPARATUS:
                    st.session_state[f"apparatus_{apparatus}"] = True
                st.rerun()
        with status_col:
            checked_count = sum(
                1 for a in REQUIRED_APPARATUS
                if st.session_state.get(f"apparatus_{a}", False)
            )
            if checked_count == len(REQUIRED_APPARATUS):
                st.success(f"✅ 全部 {len(REQUIRED_APPARATUS)} 件器材已确认")
            elif checked_count > 0:
                st.info(f"已确认 {checked_count} / {len(REQUIRED_APPARATUS)} 件器材")

        st.markdown("---")
        cols = st.columns(3)
        for i, apparatus in enumerate(REQUIRED_APPARATUS):
            cols[i % 3].checkbox(apparatus, key=f"apparatus_{apparatus}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 开始实验", use_container_width=True, type="primary"):
            st.session_state.experiment_running = True
            st.session_state.experiment_start_time = time.time()
            st.session_state.current_step_index = 0
            st.session_state.error_count = 0
            st.session_state.question_count = 0
            st.session_state.user_questions = []
            st.session_state.safety_events = []
            st.session_state.actions_log = []
            st.session_state.completed_steps = []
            st.session_state.step_started_at = st.session_state.experiment_start_time
            logger.info(f"实验开始: {experiment_type}")
            st.rerun()
    with col2:
        st.button("📖 查看实验说明", use_container_width=True)


def _show_running_experiment(experiment_type: str) -> None:
    """实验进行中的界面"""
    elapsed_time = time.time() - st.session_state.experiment_start_time
    current_step_idx = st.session_state.current_step_index
    current_step = EXPERIMENT_STEPS[min(current_step_idx, len(EXPERIMENT_STEPS) - 1)]

    # ── 顶部状态栏 ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("实验状态", "🟢 进行中")
    with col2:
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        st.metric("已用时间", f"{minutes:02d}:{seconds:02d}")
    with col3:
        st.metric("当前步骤", current_step)
    with col4:
        progress = (current_step_idx + 1) / len(EXPERIMENT_STEPS)
        st.metric("进度", f"{int(progress * 100)}%")

    # 进度条
    st.progress(progress)
    st.markdown("---")

    # ── 步骤导航 ──
    st.subheader("📋 实验步骤")
    step_cols = st.columns(len(EXPERIMENT_STEPS))
    for i, step in enumerate(EXPERIMENT_STEPS):
        with step_cols[i]:
            if i < current_step_idx:
                st.success(f"✅ {step}")
            elif i == current_step_idx:
                st.warning(f"▶ {step}")
            else:
                st.write(f"⬜ {step}")

    st.markdown("---")

    # ── 主内容区 ──
    left_col, right_col = st.columns([3, 2])

    with left_col:
        # 视频显示区域
        st.subheader("📹 实时视频")
        video_placeholder = st.empty()
        video_placeholder.info(
            "📷 摄像头视频流将在此显示\n\n"
            "系统将实时检测以下器材：\n" +
            "、".join(REQUIRED_APPARATUS[:8]) + " 等"
        )

        # 检测到的器材
        st.subheader("🎯 检测到的器材")
        detected = st.session_state.get("detected_objects", [])
        if detected:
            cols = st.columns(4)
            for i, obj in enumerate(detected):
                cols[i % 4].success(f"✅ {obj}")
        else:
            st.write("等待检测...")

    with right_col:
        # 智能指导
        st.subheader("💡 AI 实时指导")
        guidance_placeholder = st.empty()
        guidance = _get_ai_guidance(current_step, elapsed_time)
        guidance_placeholder.info(f"**当前步骤：{current_step}**\n\n{guidance}")

        # 安全警告
        safety_events = st.session_state.get("safety_events", [])
        if safety_events:
            st.subheader("⚠️ 安全提醒")
            for event in safety_events[-3:]:
                st.warning(event.get("message", ""))

        # 实验统计
        st.subheader("📊 实验统计")
        st.write(f"错误次数：{st.session_state.error_count}")
        st.write(f"提问次数：{st.session_state.question_count}")

        # 用户提问
        st.subheader("❓ 提问")
        question = st.text_input("有疑问？输入问题", key="user_question", label_visibility="collapsed")
        if st.button("发送", key="send_question") and question:
            st.session_state.question_count += 1
            st.session_state.user_questions.append(question)
            st.rerun()

    st.markdown("---")

    # ── 步骤控制 ──
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅ 上一步", use_container_width=True,
                     disabled=current_step_idx == 0):
            st.session_state.current_step_index = max(0, current_step_idx - 1)
            st.rerun()
    with col2:
        if st.button("➡ 下一步", use_container_width=True,
                     disabled=current_step_idx >= len(EXPERIMENT_STEPS) - 1):
            _record_current_step(current_step, current_step_idx)
            st.session_state.current_step_index = min(
                len(EXPERIMENT_STEPS) - 1, current_step_idx + 1
            )
            st.session_state.step_started_at = time.time()
            st.rerun()
    with col3:
        if st.button("✅ 结束实验", use_container_width=True, type="primary"):
            _finish_experiment(experiment_type, elapsed_time)

    # 自动刷新：每秒更新已用时间
    time.sleep(1)
    st.rerun()


def _get_step_hint(step: str) -> str:
    """返回步骤提示文本"""
    hints = {
        "准备器材":   "检查锥形瓶、导管、集气瓶等器材是否齐全并清洁。",
        "添加药品":   "用药勺取适量二氧化锰固体放入锥形瓶，注意不要撒落。",
        "组装装置":   "插入双孔橡胶塞，连接导管和橡胶管，夹好止水夹。",
        "检查气密性": "关闭止水夹，导管末端入水，手捂锥形瓶，观察气泡。",
        "点燃酒精灯": "用火柴点燃酒精灯，注意安全距离。",
        "收集气体":   "待气泡连续均匀冒出后，将导管伸入集气瓶底部收集。",
        "验证气体":   "将带火星的木条伸入集气瓶，复燃则证明收集到氧气。",
        "熄灭酒精灯": "用灯帽盖灭酒精灯，不可用嘴吹灭。",
        "整理器材":   "废液倒入废液缸，用洗瓶清洗器材，整理归位。",
    }
    return hints.get(step, "请按照实验规范操作。")


def _get_llm_service() -> LLMInteractionService:
    """按 Streamlit 会话复用大模型服务，以保留响应缓存。"""
    if "llm_service" not in st.session_state:
        st.session_state.llm_service = LLMInteractionService()
    return st.session_state.llm_service


def _get_ai_guidance(current_step: str, elapsed_time: float) -> str:
    """请求当前步骤的 AI 指导；服务层会在 API 不可用时自动降级。"""
    context = ExperimentContext(
        current_step=current_step,
        detected_objects=st.session_state.get("detected_objects", []),
        detected_actions=st.session_state.get("actions_log", []),
        user_questions=st.session_state.get("user_questions", []),
        error_count=st.session_state.get("error_count", 0),
        elapsed_time=elapsed_time,
        safety_events=st.session_state.get("safety_events", []),
        missing_apparatus=[
            apparatus for apparatus in REQUIRED_APPARATUS
            if apparatus not in st.session_state.get("detected_objects", [])
        ],
    )
    try:
        return _get_llm_service().provide_guidance(context)
    except Exception as e:
        logger.error(f"获取 AI 指导失败: {e}")
        return _get_step_hint(current_step)


def _record_current_step(step_name: str, step_index: int) -> None:
    """记录已完成步骤，供实验结束时的大模型表现分析使用。"""
    started_at = st.session_state.get("step_started_at") or time.time()
    ended_at = max(time.time(), started_at + 0.001)
    safety_events = st.session_state.get("safety_events", [])
    step = ExperimentStep(
        step_name=step_name,
        start_timestamp=started_at,
        end_timestamp=ended_at,
        duration=ended_at - started_at,
        pause_duration=0.0,
        anomalies=[event.get("message", "") for event in safety_events],
        sequence_order=step_index + 1,
        detected_objects=list(st.session_state.get("detected_objects", [])),
    )
    st.session_state.completed_steps.append(step)


def _finish_experiment(experiment_type: str, elapsed_time: float) -> None:
    """结束实验：计算评分、写入数据库、跳转结果页"""
    st.session_state.experiment_running = False
    st.session_state.experiment_finished = True

    current_step_idx = st.session_state.current_step_index
    current_step = EXPERIMENT_STEPS[min(current_step_idx, len(EXPERIMENT_STEPS) - 1)]
    _record_current_step(current_step, current_step_idx)

    # 使用大模型分析表现并生成建议；服务层在 API 失败时提供规则化降级结果。
    performance_analysis: Optional[PerformanceAnalysis] = None
    recommendations: List[str] = []
    try:
        performance_analysis = _get_llm_service().analyze_performance(
            experiment_steps=st.session_state.completed_steps,
            user_questions=st.session_state.user_questions,
            detected_errors=[f"记录到 {st.session_state.error_count} 次操作错误"]
            if st.session_state.error_count else [],
            total_time=elapsed_time,
            safety_events=st.session_state.safety_events,
        )
        recommendations = _get_llm_service().generate_recommendations(performance_analysis)
    except Exception as e:
        logger.error(f"大模型表现分析失败: {e}")

    # ── 计算评分 ──
    result = None
    try:
        from ..scoring.scoring_engine import ScoringEngine
        engine = ScoringEngine()
        result = engine.calculate_complete_score(
            experiment_type=experiment_type,
            actual_time=elapsed_time,
            error_count=st.session_state.error_count,
            question_count=st.session_state.question_count,
            operation_quality=(
                performance_analysis.operation_quality_score
                if performance_analysis else 0.8
            ),
            safety_compliance=(
                performance_analysis.safety_compliance
                if performance_analysis else len(st.session_state.safety_events) == 0
            ),
            safety_event_count=len(st.session_state.safety_events),
            steps_completed=st.session_state.current_step_index + 1,
            total_steps=len(EXPERIMENT_STEPS),
        )
    except Exception as e:
        logger.error(f"计算得分失败: {e}")

    # ── 写入数据库 ──
    user_id = st.session_state.get("user_id")
    if user_id and result:
        try:
            dao = ExperimentDAO()
            exp_session = dao.create_experiment_session(
                user_id=user_id,
                experiment_type=experiment_type,
                start_time=datetime.fromtimestamp(st.session_state.experiment_start_time),
            )
            dao.complete_experiment_session(
                session_id=exp_session.session_id,
                final_score=result.final_score,
                s1_score=result.s1_time_score,
                s2_score=result.s2_performance_score,
            )
            # 同时更新 percentile_rank 和 total_duration
            dao.update_experiment_session(
                session_id=exp_session.session_id,
                percentile_rank=result.percentile_rank,
                total_duration=elapsed_time,
            )
            logger.info(f"实验记录已保存: session_id={exp_session.session_id}")
        except Exception as e:
            logger.error(f"保存实验记录失败: {e}")

    # 保存结果到 session state，供结果页展示
    st.session_state.experiment_result = {
        "result": result,
        "elapsed_time": elapsed_time,
        "error_count": st.session_state.error_count,
        "question_count": st.session_state.question_count,
        "safety_event_count": len(st.session_state.safety_events),
        "experiment_type": experiment_type,
        "performance_analysis": performance_analysis,
        "recommendations": recommendations,
    }

    logger.info(f"实验结束: {experiment_type}, 用时: {elapsed_time:.1f}s")
    st.rerun()


def _show_experiment_result(experiment_type: str) -> None:
    """显示实验结果报告页"""
    data = st.session_state.get("experiment_result", {})
    result = data.get("result")
    elapsed_time = data.get("elapsed_time", 0)
    error_count = data.get("error_count", 0)
    question_count = data.get("question_count", 0)
    safety_event_count = data.get("safety_event_count", 0)
    performance_analysis = data.get("performance_analysis")
    recommendations = data.get("recommendations", [])

    st.success("🎉 实验完成！")
    st.markdown("---")
    st.subheader("📊 实验评分报告")

    if result:
        # ── 核心评分指标 ──
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最终得分", f"{result.final_score:.2f} / 2.00")
        with col2:
            st.metric("等级", result.get_grade())
        with col3:
            st.metric("时长得分 (S1)", f"{result.s1_time_score:.2f}")
        with col4:
            st.metric("表现得分 (S2)", f"{result.s2_performance_score:.2f}")

        # ── 百分位排名 ──
        st.info(f"🏆 超越了 **{result.percentile_rank:.1f}%** 的学生")

        # ── 详细统计 ──
        st.markdown("---")
        st.subheader("📋 实验详情")
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.write(f"**实验类型：** {experiment_type}")
            st.write(f"**完成时间：** {minutes} 分 {seconds} 秒")
            st.write(f"**操作错误：** {error_count} 次")
        with detail_col2:
            st.write(f"**提问次数：** {question_count} 次")
            st.write(f"**安全事件：** {safety_event_count} 次")
            details = result.calculation_details
            steps_done = details.get("steps_completed", 0)
            total_steps = details.get("total_steps", len(EXPERIMENT_STEPS))
            st.write(f"**完成步骤：** {steps_done} / {total_steps}")

        # ── 评分说明 ──
        st.markdown("---")
        st.subheader("💡 评分说明")
        st.write(result.get_summary())

        # ── 大模型评价与改进建议 ──
        if performance_analysis:
            st.markdown("---")
            st.subheader("🤖 DS Flash 表现分析")
            st.write(performance_analysis.detailed_feedback)
            st.caption(
                f"操作质量：{performance_analysis.operation_quality_score:.2f} / 1.00"
                f"｜安全规范：{'遵守' if performance_analysis.safety_compliance else '需改进'}"
            )

        suggestions = []
        if error_count > 2:
            suggestions.append("⚠️ 操作错误较多，建议多加练习基本操作规范。")
        if safety_event_count > 0:
            suggestions.append("⚠️ 存在安全事件，请注意实验安全规范。")
        if question_count > 5:
            suggestions.append("💬 提问次数较多，建议课前充分预习实验步骤。")
        if result.s1_time_score < 0.8:
            suggestions.append("⏱️ 完成时间偏长，熟悉操作流程可提升效率。")
        if not suggestions:
            suggestions.append("✅ 表现良好，继续保持！")

        st.subheader("📝 改进建议")
        for s in recommendations or suggestions:
            st.write(s)

    else:
        st.warning("评分计算失败，但实验记录已保存。")
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        st.write(f"**完成时间：** {minutes} 分 {seconds} 秒")

    st.markdown("---")
    if st.button("🔄 再次实验", type="primary", use_container_width=True):
        # 重置所有实验相关状态
        for key in ["experiment_running", "experiment_finished", "experiment_result",
                    "experiment_start_time", "current_step_index", "detected_objects",
                    "error_count", "question_count", "user_questions", "safety_events", "actions_log",
                    "completed_steps", "step_started_at"]:
            st.session_state[key] = False if key in ("experiment_running", "experiment_finished") else \
                                    None if key in ("experiment_result", "experiment_start_time", "step_started_at") else \
                                    0 if key in ("current_step_index", "error_count", "question_count") else []
        st.rerun()


# ── 历史记录页面 ──────────────────────────────────────────────────────────────

def show_history_page() -> None:
    """显示历史记录页面"""
    st.header("📚 历史记录")

    user_id = st.session_state.get("user_id")
    if not user_id:
        st.warning("请先登录")
        return

    # 筛选选项
    col1, col2 = st.columns(2)
    with col1:
        experiment_filter = st.selectbox("实验类型", ["全部"] + EXPERIMENT_TYPES)
    with col2:
        date_range = st.date_input("日期范围", [])

    st.markdown("---")

    # 从数据库查询
    try:
        dao = ExperimentDAO()
        experiments, total = dao.get_user_experiments(user_id, page_size=50)

        if not experiments:
            st.info("暂无实验记录，快去开始第一次实验吧！")
            return

        st.write(f"共 {total} 条记录")

        for exp in experiments:
            # 筛选
            if experiment_filter != "全部" and exp.experiment_type != experiment_filter:
                continue

            with st.expander(
                f"📅 {exp.start_time.strftime('%Y-%m-%d %H:%M')} — "
                f"{exp.experiment_type} | "
                f"得分: {f'{exp.final_score:.2f}' if exp.final_score is not None else '—'}"
            ):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("最终得分", f"{exp.final_score:.2f}" if exp.final_score else "—")
                with col2:
                    st.metric("S1 时长得分", f"{exp.s1_score:.2f}" if exp.s1_score else "—")
                with col3:
                    st.metric("S2 表现得分", f"{exp.s2_score:.2f}" if exp.s2_score else "—")
                with col4:
                    duration = exp.total_duration
                    if duration:
                        m, s = int(duration // 60), int(duration % 60)
                        st.metric("用时", f"{m}分{s}秒")
                    else:
                        st.metric("用时", "—")

                if exp.percentile_rank is not None:
                    st.write(f"超越了 **{exp.percentile_rank:.1f}%** 的学生")
                if exp.notes:
                    st.write(f"备注：{exp.notes}")

    except Exception as e:
        logger.error(f"获取历史记录失败: {e}")
        st.error(f"加载历史记录失败: {e}")
