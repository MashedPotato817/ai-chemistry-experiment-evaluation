#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统主入口

提供命令行接口和应用程序启动功能。
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.chemistry_lab.config import config
from src.chemistry_lab.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def setup_environment():
    """初始化系统环境"""
    # 设置日志
    setup_logger()
    
    # 创建必要目录
    config.create_directories()
    
    logger.info(f"系统初始化完成 - {config.app_name} v{config.app_version}")
    logger.info(f"项目根目录: {config.project_root}")
    logger.info(f"数据目录: {config.data_dir}")
    logger.info(f"模型目录: {config.models_dir}")
    logger.info(f"日志目录: {config.logs_dir}")


def run_streamlit():
    """启动Streamlit Web界面"""
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # 设置Streamlit参数
        sys.argv = [
            "streamlit",
            "run",
            str(project_root / "src" / "chemistry_lab" / "ui" / "main.py"),
            "--server.port", str(config.ui.port),
            "--server.address", config.ui.host,
            "--theme.base", config.ui.theme,
        ]
        
        logger.info(f"启动Streamlit服务器: http://{config.ui.host}:{config.ui.port}")
        stcli.main()
        
    except ImportError:
        logger.error("Streamlit未安装，请运行: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        logger.error(f"启动Streamlit失败: {e}")
        sys.exit(1)


def run_fastapi():
    """启动FastAPI服务器"""
    try:
        import uvicorn
        
        logger.info(f"启动FastAPI服务器: http://{config.ui.host}:{config.ui.port}")
        uvicorn.run(
            "src.chemistry_lab.api.main:app",
            host=config.ui.host,
            port=config.ui.port,
            reload=config.debug,
            log_level=config.log_level.lower(),
        )
        
    except ImportError:
        logger.error("FastAPI或uvicorn未安装，请运行: pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"启动FastAPI失败: {e}")
        sys.exit(1)


def run_tests():
    """运行测试套件"""
    try:
        import pytest
        
        logger.info("运行测试套件...")
        exit_code = pytest.main([
            "tests/",
            "-v",
            "--cov=src/chemistry_lab",
            "--cov-report=html",
            "--cov-report=term-missing",
        ])
        
        sys.exit(exit_code)
        
    except ImportError:
        logger.error("pytest未安装，请运行: pip install pytest pytest-cov")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="大模型驱动的智能化学实验熟练度评估实时交互系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s streamlit          # 启动Streamlit Web界面
  %(prog)s fastapi            # 启动FastAPI服务器
  %(prog)s test               # 运行测试套件
  %(prog)s --version          # 显示版本信息
        """
    )
    
    parser.add_argument(
        "command",
        choices=["streamlit", "fastapi", "test"],
        help="要执行的命令"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {config.app_version}"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    
    # 设置调试模式
    if args.debug:
        config.debug = True
        config.log_level = "DEBUG"
    
    # 初始化环境
    setup_environment()
    
    # 执行命令
    if args.command == "streamlit":
        run_streamlit()
    elif args.command == "fastapi":
        run_fastapi()
    elif args.command == "test":
        run_tests()


if __name__ == "__main__":
    main()