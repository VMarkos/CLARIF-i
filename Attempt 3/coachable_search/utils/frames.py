"""
Frame layouts and organization for the coaching interface.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Set, List, Tuple

from .widgets import TableWidget, TextWidget, CanvasWidget

class StateSpaceFrame(ttk.LabelFrame):
    """Frame for displaying state space information."""
    
    def __init__(self, parent):
        """Initialize the state space frame."""
        super().__init__(parent, text="State Space", padding="5")
        self.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.text = TextWidget(self, height=10, width=40)
    
    def update(self, state_space: Dict[str, Set[str]]):
        """Update the state space display."""
        self.text.clear()
        if state_space:
            for var, domain in state_space.items():
                self.text.insert(f"{var}: {', '.join(domain)}\n")
        else:
            self.text.insert("No state space defined")

class StatesFrame(ttk.LabelFrame):
    """Frame for displaying current states."""
    
    def __init__(self, parent):
        """Initialize the states frame."""
        super().__init__(parent, text="Current States", padding="5")
        self.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.text = TextWidget(self, height=10, width=40)
    
    def update(self, start_state: Dict[str, str], goal_state: Dict[str, str]):
        """Update the states display."""
        self.text.clear()
        if start_state and goal_state:
            self.text.insert("Start State:\n")
            self.text.insert(self._format_state(start_state))
            self.text.insert("\n\nGoal State:\n")
            self.text.insert(self._format_state(goal_state))
        else:
            self.text.insert("States not defined")
    
    def _format_state(self, state: Dict[str, str]) -> str:
        """Format a state for display."""
        return '\n'.join(f"{k}: {v}" for k, v in state.items())

class PerformanceFrame(ttk.LabelFrame):
    """Frame for displaying performance history."""
    
    def __init__(self, parent):
        """Initialize the performance frame."""
        super().__init__(parent, text="Performance History", padding="5")
        self.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.canvas = CanvasWidget(self, height=200)
    
    def update(self, performance_history: List[float]):
        """Update the performance display."""
        self.canvas.clear()
        
        if not performance_history:
            return
        
        # Calculate dimensions
        width = self.canvas.canvas.winfo_width()
        height = self.canvas.canvas.winfo_height()
        padding = 20
        
        # Draw axes
        self.canvas.draw_axes(width, height, padding)
        
        # Draw performance line
        if len(performance_history) > 1:
            x_step = (width - 2*padding) / max(1, len(performance_history) - 1)
        else:
            x_step = width - 2*padding  # Single point case
            
        points = []
        for i, value in enumerate(performance_history):
            x = padding + i * x_step
            y = height - padding - value * (height - 2*padding)
            points.extend([x, y])
        
        if points:
            self.canvas.draw_line(points)

class TraceFrame(ttk.LabelFrame):
    """Frame for displaying reasoning traces."""
    
    def __init__(self, parent):
        """Initialize the trace frame."""
        super().__init__(parent, text="Current Reasoning Trace", padding="5")
        self.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.text = TextWidget(self, height=30, width=40)
    
    def update(self, trace: List[Tuple[Dict[str, str], str]]):
        """Update the trace display."""
        self.text.clear()
        if trace:
            self.text.insert(str(trace))
        else:
            self.text.insert("No traces available")
        self.text.set_readonly()

class RulesFrame(ttk.LabelFrame):
    """Frame for displaying rules."""
    
    def __init__(self, parent, title: str):
        """Initialize the rules frame."""
        super().__init__(parent, text=title, padding="5")
        self.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.table = TableWidget(self)
    
    def update(self, rules: List[Rule]):
        """Update the rules display."""
        self.table.update(rules)

class AdviceFrame(ttk.LabelFrame):
    """Frame for displaying advice."""
    
    def __init__(self, parent):
        """Initialize the advice frame."""
        super().__init__(parent, text="Current Advice", padding="5")
        self.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.text = TextWidget(self, height=10, width=40)
    
    def update(self, feedback_rules: List[Rule]):
        """Update the advice display."""
        self.text.clear()
        if feedback_rules:
            advice_text = '\n'.join(
                f"Rule {i+1}: If {self._format_condition(rule.condition)} then {self._format_preference(rule.preference)}"
                for i, rule in enumerate(feedback_rules)
            )
            self.text.insert(advice_text)
        else:
            self.text.insert("No advice needed")
        self.text.set_readonly()
    
    def _format_condition(self, condition: Dict[str, str]) -> str:
        """Format a condition for display."""
        return ', '.join(f"{k}={v}" for k, v in condition.items())
    
    def _format_preference(self, preference: Dict[str, str]) -> str:
        """Format a preference for display."""
        return ', '.join(f"{k}={v}" for k, v in preference.items()) 