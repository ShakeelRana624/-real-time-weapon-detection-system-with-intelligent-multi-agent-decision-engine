"""
Integrated Gun Detection System

A comprehensive real-time threat detection system combining
YOLO-based computer vision with agent-based decision making.
"""

__version__ = "1.0.0"
__author__ = "Hybrid Decision Engine Team"
__description__ = "Real-time gun detection with intelligent threat assessment"

from .agent_based_decision_engine import AgentBasedDecisionEngine
from .integrated_gun_detection_system import IntegratedGunDetectionSystem
from .hybrid_decision_engine import DecisionEngine

__all__ = [
    'AgentBasedDecisionEngine',
    'IntegratedGunDetectionSystem', 
    'DecisionEngine'
]
