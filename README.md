# 🧪 智能化学实验熟练度评估系统

大模型驱动的智能化学实验评估平台

## 📖 项目简介

这是一个基于 Python 的智能教育平台，集成了计算机视觉、大模型技术和实时交互功能，为化学实验学习者提供个性化的熟练度评估和智能指导。

### ✨ 核心功能

- 🔐 **用户管理** - 登录注册、会话管理
- 📊 **智能评分** - 双重评分机制（时长 + 表现）
- 🎥 **视频检测** - YOLOv8 实时目标检测（自定义模型 `fs_s_best.pt`）
- 🕐 **时序分析** - 实验步骤自动推断、动作识别、安全检测
- 🤖 **AI 指导** - 大模型驱动的实时交互；实验中提供步骤指导、回答提问，结束后生成表现分析与改进建议（支持 Kimi / Qwen / DeepSeek / OpenAI）
- 📈 **数据分析** - 实验历史记录和统计

### 🎯 评分算法

系统采用双重评分机制：

- **S1 时长得分** (70% 权重): 基于正态分布 `f(t) = 2 - 2Φ((μ-t)/σ)`
- **S2 表现得分** (30% 权重): 综合操作质量、错误次数、安全规范等
- **最终得分**: `S = 0.7 × S1 + 0.3 × S2`（满分 2.0）
- **百分位排名**: 基于最终得分的正态分布估计，反映超越同类学生的比例

## 🚀 快速开始

### 前置要求

- Python 3.9 或更高版本
- pip 包管理器
- Windows 上建议使用 [python.org](https://www.python.org/downloads/) 的独立 CPython，避免 Anaconda 环境与 PyTorch 原生 DLL 冲突

### 1️⃣ 克隆项目

```bash
git clone <your-repo-url>
cd setup
```

### 2️⃣ 创建虚拟环境并安装依赖（Windows 推荐）

```powershell
# 用独立 CPython 创建虚拟环境；将路径替换为本机 python.exe 路径
& "C:\Users\<用户名>\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv-cpython

# 激活 UTF-8 模式，避免带中文注释的 requirements.txt 被 GBK 错误解析
$env:PYTHONUTF8 = "1"

# 先安装已验证的 CPU 版 PyTorch（适合没有 CUDA 或只需演示的场景）
.\.venv-cpython\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0

# 安装项目其余依赖
.\.venv-cpython\Scripts\python.exe -m pip install -r requirements.txt
```

> 如需 GPU 推理，请根据显卡与驱动从 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择对应 CUDA wheel；CPU 模式请保持 `YOLO_DEVICE=cpu`。

### 3️⃣ 配置大模型 API

编辑 `.env` 文件，填入你的 API 密钥和模型信息：

```dotenv
LLM_API_KEY=your-api-key-here
LLM_API_BASE=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-v4-flash
DEBUG=false
YOLO_DEVICE=cpu
```

> 💡 **提示**: 不配置 API 密钥也能运行，系统会使用内置降级响应。

支持的模型服务商见下方 [大模型接入](#-大模型接入) 章节。

### 4️⃣ 启动应用

**方式一：使用启动脚本（推荐）**
```powershell
.\.venv-cpython\Scripts\python.exe run.py
```

**方式二：直接运行 Streamlit**
```powershell
.\.venv-cpython\Scripts\python.exe -m streamlit run app.py
```

**方式三：运行功能演示**
```bash
python demo.py
```

### 5️⃣ 访问应用

打开浏览器访问：**http://localhost:8501**

## 🤖 大模型接入

系统使用 OpenAI 兼容接口，只需修改 `.env` 中三个参数即可切换任意兼容模型，**无需改代码**。

### Kimi（月之暗面）

```dotenv
LLM_API_KEY=sk-your-moonshot-key
LLM_API_BASE=https://api.moonshot.cn/v1
LLM_MODEL_NAME=moonshot-v1-8k
```

可选模型：`moonshot-v1-8k` / `moonshot-v1-32k` / `moonshot-v1-128k`

### 通义千问（Qwen）

```dotenv
LLM_API_KEY=sk-your-dashscope-key
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen-plus
```

可选模型：`qwen-turbo` / `qwen-plus` / `qwen-max` / `qwen-vl-plus`（多模态）

### DeepSeek

```dotenv
LLM_API_KEY=sk-your-deepseek-key
LLM_API_BASE=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-v4-flash
```

`deepseek-v4-flash` 已接入实验页面：进入每一步时生成指导；发送问题时直接回答；结束实验时生成表现分析和 3～5 条改进建议。API 不可用时会自动回退到内置规则提示。

### OpenAI

```dotenv
LLM_API_KEY=sk-your-openai-key
LLM_API_BASE=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o
```

## 📁 项目结构

```
setup/
├── src/chemistry_lab/          # 主要源代码
│   ├── config/                 # 配置管理（读取 .env）
│   ├── database/               # 数据库和 DAO
│   ├── user_management/        # 用户认证和会话
│   ├── detection/              # YOLO 检测、时序分析、视频处理
│   ├── scoring/                # 评分引擎
│   ├── llm/                    # 大模型交互服务
│   ├── ui/                     # Streamlit 界面
│   └── utils/                  # 工具函数
├── tests/                      # 测试代码
│   ├── property_tests/         # 基于属性的测试
│   └── unit_tests/             # 单元测试
├── model_tests/                # YOLO 模型测试
├── models/                     # 模型文件（fs_s_best.pt）
├── logs/                       # 日志文件（自动创建）
├── data/                       # 数据目录（自动创建）
├── app.py                      # Streamlit 主应用
├── run.py                      # 启动脚本
├── demo.py                     # 功能演示脚本
├── requirements.txt            # 依赖列表
├── .env                        # 环境变量配置
└── README.md                   # 项目文档
```

## 💻 使用示例

### 运行功能演示

```bash
python demo.py
```

这会展示：
- ✅ 评分引擎的三个示例场景
- ✅ 大模型服务的实时指导（需配置 API Key）
- ✅ 系统核心模块的工作流程

### 评分引擎示例

```python
from chemistry_lab.scoring import ScoringEngine

engine = ScoringEngine()

result = engine.calculate_complete_score(
    experiment_type="双氧水制氧气",
    actual_time=280.0,        # 实际用时（秒）
    error_count=2,            # 错误次数
    question_count=3,         # 提问次数
    operation_quality=0.85,   # 操作质量 (0-1)
    safety_compliance=True    # 是否遵守安全规范
)

print(f"最终得分: {result.final_score:.2f} / 2.0")
print(f"百分位排名: 超越 {result.percentile_rank:.1f}% 的学生")
```

## 🧪 开发指南

### 运行测试

```bash
# 运行所有测试
pytest

# 运行属性测试
pytest tests/property_tests/

# 运行单元测试
pytest tests/unit_tests/

# 查看测试覆盖率
pytest --cov=src/chemistry_lab
```

### 代码质量

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

## ⚙️ 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.9+ |
| Web 框架 | Streamlit |
| 数据库 | SQLite + SQLAlchemy |
| 目标检测 | Ultralytics YOLOv8（自定义模型 fs_s_best.pt） |
| 图像处理 | OpenCV |
| 大模型 | OpenAI 兼容接口（Kimi / Qwen / DeepSeek / OpenAI） |
| 科学计算 | NumPy, SciPy, Pandas |
| 测试框架 | Pytest, Hypothesis |

## ⚠️ 注意事项

1. **YOLO 模型**: 使用自定义训练模型 `models/fs_s_best.pt`，针对气体制备与收集实验的19类器材优化
2. **大模型 API**: 可选配置，不配置也能运行（使用降级响应）；支持任何 OpenAI 兼容接口
3. **摄像头权限**: 实时检测功能需要摄像头访问权限
4. **性能建议**: `.env` 中设置 `YOLO_DEVICE=cuda` 可启用 GPU 加速（≥15 FPS）

## 🐛 常见问题

### 1. 依赖安装失败？
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 大模型 API 不可用？
检查 `.env` 中以下三项是否正确填写：
```dotenv
LLM_API_KEY=your-actual-key
LLM_API_BASE=https://api.moonshot.cn/v1
LLM_MODEL_NAME=moonshot-v1-8k
```

### 3. 数据库错误？
```bash
# Windows
del chemistry_lab.db
# Linux/Mac
rm chemistry_lab.db

python run.py
```

### 4. 端口被占用？
```bash
streamlit run app.py --server.port 8502
```

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过 Issue 联系我们。
