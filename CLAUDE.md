# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

智能化学实验熟练度评估系统 - 基于大模型和YOLOv8的实时交互评估平台。

## 常用命令

```bash
# 启动应用
python run.py                    # 推荐方式
streamlit run app.py             # 直接运行 Streamlit

# 运行测试
pytest                           # 运行所有测试（覆盖率 ≥80%）
pytest tests/unit_tests/         # 单元测试
pytest tests/property_tests/     # 基于属性的测试（Hypothesis）
pytest -m "not slow"             # 跳过慢速测试
pytest -k "test_name"            # 运行指定测试

# 代码质量
black src/ tests/                # 格式化
flake8 src/ tests/               # lint
mypy src/                        # 类型检查
```

## 架构

```
src/chemistry_lab/
├── config/       # Pydantic Settings 配置（读取 .env）
├── database/     # SQLAlchemy + SQLite，DAO 模式
├── user_management/  # JWT 认证、会话管理
├── detection/    # YOLOv8 检测（fs_s_best.pt）、时序分析、视频处理
├── scoring/      # 评分引擎：S1 时长(70%) + S2 表现(30%)
├── llm/          # OpenAI 兼容接口（Kimi/Qwen/DeepSeek/OpenAI）
├── ui/           # Streamlit 页面
└── utils/        # 日志、异常、工具函数
```

## 关键设计决策

- **配置**: 所有配置通过 `.env` 环境变量管理，使用 `pydantic-settings` 自动加载
- **LLM**: 使用 OpenAI 兼容接口，切换服务商只需修改 `.env` 中三个参数
- **评分算法**: `S = 0.7 × S1 + 0.3 × S2`，满分 2.0，S1 基于正态分布，S2 综合操作质量
- **YOLO 模型**: 自定义训练模型 `models/fs_s_best.pt`，19 类气体制备实验器材
- **测试**: 同时使用 pytest 单元测试和 Hypothesis 属性测试

## 环境配置

复制 `.env.example` 为 `.env`，必须配置 `LLM_API_KEY`、`LLM_API_BASE`、`LLM_MODEL_NAME` 三个参数才能使用 AI 功能。

## 依赖

```bash
pip install -r requirements.txt
```

核心依赖：Streamlit、Ultralytics YOLOv8、SQLAlchemy、OpenAI、SciPy、Pydantic
