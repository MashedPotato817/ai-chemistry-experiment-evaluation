#!/usr/bin/env python3
"""
核心功能演示脚本

展示评分引擎和大模型服务的使用
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chemistry_lab.scoring import ScoringEngine
from chemistry_lab.llm import LLMInteractionService, ExperimentContext
from chemistry_lab.detection.models import Action


def demo_scoring_engine():
    """演示评分引擎"""
    print("=" * 60)
    print("📊 评分引擎演示")
    print("=" * 60)
    print()
    
    # 初始化评分引擎
    engine = ScoringEngine()
    
    # 示例1: 快速完成的实验
    print("示例1: 快速完成的实验")
    print("-" * 40)
    result1 = engine.calculate_complete_score(
        experiment_type="双氧水制氧气",
        actual_time=240.0,  # 4分钟
        error_count=1,
        question_count=2,
        operation_quality=0.90,
        safety_compliance=True
    )
    
    print(f"实际用时: 240秒 (4分钟)")
    print(f"错误次数: 1")
    print(f"提问次数: 2")
    print(f"操作质量: 0.90")
    print()
    print(f"✅ S1时长得分: {result1.s1_time_score:.3f} / 2.0")
    print(f"✅ S2表现得分: {result1.s2_performance_score:.3f} / 2.0")
    print(f"🎯 最终得分: {result1.final_score:.3f} / 2.0")
    print(f"📈 百分位排名: 超越 {result1.percentile_rank:.1f}% 的学生")
    print()
    
    # 示例2: 较慢但操作规范的实验
    print("示例2: 较慢但操作规范的实验")
    print("-" * 40)
    result2 = engine.calculate_complete_score(
        experiment_type="双氧水制氧气",
        actual_time=360.0,  # 6分钟
        error_count=0,
        question_count=1,
        operation_quality=0.95,
        safety_compliance=True
    )
    
    print(f"实际用时: 360秒 (6分钟)")
    print(f"错误次数: 0")
    print(f"提问次数: 1")
    print(f"操作质量: 0.95")
    print()
    print(f"✅ S1时长得分: {result2.s1_time_score:.3f} / 2.0")
    print(f"✅ S2表现得分: {result2.s2_performance_score:.3f} / 2.0")
    print(f"🎯 最终得分: {result2.final_score:.3f} / 2.0")
    print(f"📈 百分位排名: 超越 {result2.percentile_rank:.1f}% 的学生")
    print()
    
    # 示例3: 有多个错误的实验
    print("示例3: 有多个错误的实验")
    print("-" * 40)
    result3 = engine.calculate_complete_score(
        experiment_type="双氧水制氧气",
        actual_time=420.0,  # 7分钟
        error_count=5,
        question_count=8,
        operation_quality=0.60,
        safety_compliance=False
    )
    
    print(f"实际用时: 420秒 (7分钟)")
    print(f"错误次数: 5")
    print(f"提问次数: 8")
    print(f"操作质量: 0.60")
    print(f"安全规范: 未遵守")
    print()
    print(f"✅ S1时长得分: {result3.s1_time_score:.3f} / 2.0")
    print(f"✅ S2表现得分: {result3.s2_performance_score:.3f} / 2.0")
    print(f"🎯 最终得分: {result3.final_score:.3f} / 2.0")
    print(f"📈 百分位排名: 超越 {result3.percentile_rank:.1f}% 的学生")
    print()
    
    print("=" * 60)
    print()


def demo_llm_service():
    """演示大模型服务"""
    print("=" * 60)
    print("🤖 大模型服务演示")
    print("=" * 60)
    print()

    print("注意: 此演示需要在 .env 中配置 LLM_API_KEY、LLM_API_BASE、LLM_MODEL_NAME")
    print("如果未配置 API 密钥，将使用内置降级响应")
    print()
    
    try:
        # 初始化服务（会尝试从环境变量读取API密钥）
        llm_service = LLMInteractionService()
        
        # 创建实验上下文
        context = ExperimentContext(
            current_step="加液",
            detected_objects=["滴管", "双氧水瓶", "锥形瓶"],
            detected_actions=[
                Action(
                    action_type="取样",
                    start_time=0.0,
                    end_time=15.0,
                    involved_objects=["滴管", "双氧水瓶"],
                    confidence=0.9
                )
            ],
            user_questions=["需要加多少双氧水？"],
            error_count=1,
            elapsed_time=120.0,
            safety_events=[]
        )
        
        print("实验上下文:")
        print(f"  当前步骤: {context.current_step}")
        print(f"  已用时间: {context.elapsed_time}秒")
        print(f"  错误次数: {context.error_count}")
        print(f"  提问次数: {len(context.user_questions)}")
        print()
        
        # 获取指导
        print("正在获取AI指导...")
        guidance = llm_service.provide_guidance(context)
        print()
        print("💡 AI指导:")
        print(guidance)
        print()
        
    except Exception as e:
        print(f"⚠️  大模型服务不可用: {e}")
        print("使用降级响应:")
        print("💡 缓慢加入液体，避免溅出。注意观察反应现象。")
        print()
    
    print("=" * 60)
    print()


def main():
    """主函数"""
    # 显示当前 LLM 配置
    try:
        from chemistry_lab.config.config import config as app_config
        llm_cfg = app_config.llm
        llm_info = f"{llm_cfg.model_name} @ {llm_cfg.api_base}"
    except Exception:
        llm_info = "未知"

    print()
    print("🧪 智能化学实验熟练度评估系统 - 核心功能演示")
    print(f"   当前 LLM: {llm_info}")
    print()

    # 演示评分引擎
    demo_scoring_engine()

    # 演示大模型服务
    demo_llm_service()

    print("✅ 演示完成！")
    print()
    print("提示:")
    print("  - 评分算法位于: src/chemistry_lab/scoring/scoring_engine.py")
    print("  - 大模型服务位于: src/chemistry_lab/llm/llm_service.py")
    print("  - 切换模型: 修改 .env 中的 LLM_API_KEY / LLM_API_BASE / LLM_MODEL_NAME")
    print("  - 启动Web应用: python run.py")
    print()


if __name__ == "__main__":
    main()
