"""
Convergence tracking utilities for the coachable search framework.
"""

from collections import deque

class ConvergenceTracker:
    """
    Tracks the convergence of the learner's performance.
    
    Attributes:
        correct_inferences: Deque of recent inference results
        total_inferences: Total number of inferences made
    """
    
    def __init__(self):
        """Initialize the convergence tracker."""
        self.correct_inferences = deque(maxlen=100)
        self.total_inferences = 0
    
    def update(self, is_correct: bool):
        """
        Update the tracker with a new inference result.
        
        Args:
            is_correct: Whether the inference was correct
        """
        self.correct_inferences.append(is_correct)
        self.total_inferences += 1
    
    def get_convergence_rate(self) -> float:
        """
        Get the current convergence rate.
        
        Returns:
            Float between 0 and 1 representing the success rate
        """
        if not self.correct_inferences:
            return 0.0
        return sum(self.correct_inferences) / len(self.correct_inferences)
    
    def is_converged(self, threshold: float = 0.95, window_size: int = 10) -> bool:
        """
        Check if the learner has converged.
        
        Args:
            threshold: Minimum success rate required for convergence
            window_size: Number of recent inferences to consider
            
        Returns:
            True if converged, False otherwise
        """
        if len(self.correct_inferences) < window_size:
            return False
        recent_correct = sum(list(self.correct_inferences)[-window_size:])
        return recent_correct / window_size >= threshold 