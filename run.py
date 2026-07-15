#!/usr/bin/env python3
"""
启动脚本

快速启动智能化学实验评估系统
"""

import subprocess
import sys
from pathlib import Path


def main():
    """主函数"""
    print("=" * 60)
    print("🧪 智能化学实验熟练度评估系统")
    print("=" * 60)
    print()
    
    # 检查依赖
    print("检查依赖...")
    try:
        import streamlit
        import ultralytics
        import openai
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)
    
    print()
    print("启动Streamlit应用...")
    print("=" * 60)
    print()
    
    # 启动Streamlit
    app_path = Path(__file__).parent / "app.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=localhost"
    ])


if __name__ == "__main__":
    main()
