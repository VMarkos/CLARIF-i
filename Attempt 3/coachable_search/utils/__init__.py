"""
Utility components for the coachable search framework.
"""

from .interface import CoachingInterface
from .convergence import ConvergenceTracker
from .scenarios import ScenarioManager, DEMO_SCENARIOS

__all__ = ['CoachingInterface', 'ConvergenceTracker', 'ScenarioManager', 'DEMO_SCENARIOS'] 