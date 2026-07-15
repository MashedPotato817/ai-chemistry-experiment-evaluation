"""
智能化学实验熟练度评估系统 - Streamlit主应用

大模型驱动的实时交互评估系统
"""

import streamlit as st
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chemistry_lab.ui.pages import (
    show_login_page,
    show_register_page,
    show_main_page,
    show_experiment_page,
    show_history_page
)
from chemistry_lab.utils.logger import get_logger

logger = get_logger(__name__)


def init_session_state():
    """初始化会话状态"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'login'


def main():
    """主函数"""
    # 页面配置
    st.set_page_config(
        page_title="智能化学实验评估系统",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化会话状态
    init_session_state()
    
    # 自定义CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 1rem 0;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 显示标题
    st.markdown('<div class="main-header">🧪 智能化学实验熟练度评估系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">大模型驱动的实时交互评估平台</div>', unsafe_allow_html=True)
    
    # 根据登录状态显示不同页面
    if not st.session_state.logged_in:
        # 未登录：显示登录/注册页面
        tab1, tab2 = st.tabs(["登录", "注册"])
        
        with tab1:
            show_login_page()
        
        with tab2:
            show_register_page()
    
    else:
        # 已登录：显示主界面
        # 侧边栏导航
        with st.sidebar:
            st.markdown(f"### 👤 欢迎, {st.session_state.username}!")
            st.markdown("---")
            
            # 导航菜单
            page = st.radio(
                "导航",
                ["主页", "开始实验", "历史记录", "退出登录"],
                key="navigation"
            )
            
            if page == "退出登录":
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.session_state.username = None
                st.rerun()
        
        # 显示对应页面
        if page == "主页":
            show_main_page()
        elif page == "开始实验":
            show_experiment_page()
        elif page == "历史记录":
            show_history_page()


if __name__ == "__main__":
    main()
