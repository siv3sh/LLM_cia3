#!/usr/bin/env python3
"""
Setup script for Multi-Agent Attrition Analysis System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multi-agent-attrition-system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive, AI-powered system for analyzing employee attrition using multiple specialized agents",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multi-agent-attrition-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "attrition-system=streamlit_integrated:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.env"],
    },
    keywords="ai, machine-learning, attrition-analysis, multi-agent, langchain, groq, streamlit",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/multi-agent-attrition-system/issues",
        "Source": "https://github.com/yourusername/multi-agent-attrition-system",
        "Documentation": "https://github.com/yourusername/multi-agent-attrition-system#readme",
    },
)
