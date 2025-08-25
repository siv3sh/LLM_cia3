"""
Multi-Agent System for Attrition Analysis
"""

from .base_agent import BaseAgent, AgentMessage, AgentState
from .coordinator_agent import CoordinatorAgent
from .data_agent import DataAgent
from .analysis_agent import AnalysisAgent
from .prediction_agent import PredictionAgent
from .insight_agent import InsightAgent
from .chat_agent import ChatAgent

__all__ = [
    'BaseAgent',
    'AgentMessage', 
    'AgentState',
    'CoordinatorAgent',
    'DataAgent',
    'AnalysisAgent',
    'PredictionAgent',
    'InsightAgent',
    'ChatAgent'
]

__version__ = "1.0.0"
__author__ = "Multi-Agent System Team"
