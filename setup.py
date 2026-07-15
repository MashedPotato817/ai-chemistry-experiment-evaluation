#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型驱动的智能化学实验熟练度评估实时交互系统
AI-Driven Chemistry Lab Assessment Real-time Interactive System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-chemistry-lab-assessment",
    version="0.1.0",
    author="Chemistry Lab Assessment Team",
    description="大模型驱动的智能化学实验熟练度评估实时交互系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    include_package_data=True,
)