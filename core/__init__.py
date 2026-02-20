"""
Integrated Gun Detection System

A comprehensive real-time threat detection system combining
YOLO-based computer vision with agent-based decision making.
"""

__version__ = "1.0.0"
__author__ = "AI Security Systems"
__description__ = "Real-time weapon detection with intelligent threat assessment"

from .integrated_gun_detection_system import IntegratedGunDetectionSystem
from .run_system import main as run_system_main

__all__ = [
    'IntegratedGunDetectionSystem', 
    'run_system_main'
]
