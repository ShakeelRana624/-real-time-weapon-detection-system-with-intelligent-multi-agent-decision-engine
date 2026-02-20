"""
Detection Module

This module contains object detection and tracking components
for weapons, humans, and activities.
"""

from .activity_detection import ActivityDetector
from .human_tracker import HumanTracker

__all__ = ['ActivityDetector', 'HumanTracker']
