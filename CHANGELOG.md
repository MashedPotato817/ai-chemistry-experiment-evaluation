# 更新日志

## [0.2.0] - 2026-04-17

### 🐛 Bug 修复

- **配置读取修复**: `LLMConfig`、`DatabaseConfig`、`YOLOConfig`、`UIConfig` 子配置类缺少 `env_file` 声明，导致 `.env` 文件中的配置无法被读取，只能依赖系统环境变量。现已全部补充 `env_file = ".env"` 配置。
- **LLM 初始化修复**: `LLMInteractionService.__init__` 未从 `app_config.llm` 读取配置，导致即使 `.env` 中配置了 API Key 也无法生效。现已修复为优先读取 `.env` 配置，显式传参优先级更高。
- **ExperimentContext 参数修复**: `demo.py` 中创建 `ExperimentContext` 时缺少必填字段 `detected_objects` 和 `safety_events`，导致运行报错。

### ✨ 新增功能

- **多模型支持**: LLM 服务现支持任何 OpenAI 兼容接口，只需修改 `.env` 即可切换 Kimi、Qwen、DeepSeek、OpenAI 等模型，无需改代码。
- **百分位排名修正**: 原百分位基于完成时间计算，语义混乱（用时短排名高，但显示"前X%"含义相反）。现改为基于最终得分计算，"超越 X% 的学生"语义清晰准确。
- **LLM 初始化日志增强**: 初始化时记录模型名称和 API 地址，便于排查配置问题。

### 📚 文档更新

- `README.md`: 新增大模型接入章节，列出 Kimi / Qwen / DeepSeek / OpenAI 的配置方式；更新技术栈说明；修正评分算法描述；修正常见问题中的数据库删除命令（Windows 兼容）；移除多余的 QUICKSTART.md 引用
- `QUICKSTART.md`: 已删除（内容与 README.md 高度重叠，合并后移除）
- `demo_test.md`: 同步更新为最新运行结果

---

## [0.1.0] - 2026-03-05

### ✨ 新增功能

- 🔐 用户认证系统（登录、注册、会话管理）
- 📊 双重评分引擎（S1 时长得分 + S2 表现得分）
- 🎥 YOLOv8 目标检测集成（自定义模型 fs_s_best.pt，19类实验器材）
- 🕐 时序分析器（步骤推断、动作识别、安全检测）
- 🤖 大模型集成（支持降级响应）
- 📈 实验历史记录和统计
- 🎨 Streamlit Web 界面
- 🧪 功能演示脚本（demo.py）

### 🔧 技术实现

- SQLite 数据库 + SQLAlchemy ORM
- Pydantic v2 配置管理
- 基于属性的测试（Hypothesis）
- 日志系统（Loguru）
- 环境变量配置（.env）

### 🐛 已知问题

- 摄像头实时检测功能需要进一步优化
- 部分测试用例需要完善

### 🔮 计划功能

- [ ] 支持更多实验类型
- [ ] 实验视频回放功能
- [ ] 多语言支持
- [ ] 移动端适配
- [ ] 导出实验报告（PDF）
- [ ] 实验数据可视化增强

---

## 版本说明

版本号格式：`主版本号.次版本号.修订号`

- **主版本号**: 重大架构变更或不兼容的 API 修改
- **次版本号**: 新增功能，向后兼容
- **修订号**: Bug 修复和小改进
