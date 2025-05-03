"""
Main interface class for the coaching system.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, List, Any
import os

from ..core.rule import Rule
from .scenarios import ScenarioManager, DEMO_SCENARIOS

class CoachingInterface:
    """Interactive interface for visualizing the coaching process."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Coachable Search Interface")
        self.root.geometry("1200x800")
        
        # Initialize scenario manager
        self.scenario_manager = ScenarioManager()
        self.current_data = None
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create main frames
        self.create_main_frames()
        
        # Initialize state
        self.performance_history = []
    
    def create_menu_bar(self):
        """Create the menu bar with File menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        
        file_menu.add_command(label="New Scenario", command=self.new_scenario)
        file_menu.add_command(label="Open Scenario...", command=self.open_scenario)
        file_menu.add_command(label="Save Scenario", command=self.save_scenario)
        file_menu.add_command(label="Save Scenario As...", command=self.save_scenario_as)
        file_menu.add_separator()
        
        # Demo scenarios submenu
        demo_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Load Demo", menu=demo_menu)
        for scenario_name in DEMO_SCENARIOS:
            demo_menu.add_command(
                label=scenario_name.replace('_', ' ').title(),
                command=lambda name=scenario_name: self.load_demo_scenario(name)
            )
        
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
    
    def create_main_frames(self):
        """Create the main frames for the interface."""
        # Left panel for rules and context
        left_frame = ttk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Rules frame
        rules_frame = ttk.LabelFrame(left_frame, text="Rules")
        rules_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Learner rules
        self.learner_rules_table = self.create_table(rules_frame, "Learner Rules")
        self.learner_rules_table.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Coach rules
        self.coach_rules_table = self.create_table(rules_frame, "Coach Rules")
        self.coach_rules_table.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Context frame
        context_frame = ttk.LabelFrame(left_frame, text="Context")
        context_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.context_text = tk.Text(context_frame, height=5)
        self.context_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel for advice and performance
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Advice frame
        advice_frame = ttk.LabelFrame(right_frame, text="Advice")
        advice_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.advice_text = tk.Text(advice_frame, height=5)
        self.advice_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Performance frame
        performance_frame = ttk.LabelFrame(right_frame, text="Performance History")
        performance_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.performance_canvas = tk.Canvas(performance_frame)
        self.performance_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.continue_button = ttk.Button(button_frame, text="Continue", command=self.on_continue)
        self.continue_button.pack(side=tk.RIGHT, padx=5)
    
    def create_table(self, parent, title):
        """Create a table widget."""
        frame = ttk.LabelFrame(parent, text=title)
        
        # Create treeview
        tree = ttk.Treeview(frame, columns=("Name", "Condition", "Action", "Priority"), show="headings")
        tree.heading("Name", text="Name")
        tree.heading("Condition", text="Condition")
        tree.heading("Action", text="Action")
        tree.heading("Priority", text="Priority")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        return tree
    
    def update_interface(self, data: Dict[str, Any]):
        """Update the interface with new data."""
        self.current_data = data
        
        # Update rules tables
        self.update_table(self.learner_rules_table, data.get('learner_rules', []))
        self.update_table(self.coach_rules_table, data.get('coach_rules', []))
        
        # Update context
        context_text = f"Start State: {data.get('start_state', {})}\n"
        context_text += f"Goal State: {data.get('goal_state', {})}\n"
        context_text += f"Current State: {data.get('current_state', {})}"
        self.context_text.delete(1.0, tk.END)
        self.context_text.insert(tk.END, context_text)
        
        # Update advice
        advice = data.get('advice', '')
        self.advice_text.delete(1.0, tk.END)
        self.advice_text.insert(tk.END, advice)
        
        # Update performance history
        self.performance_history = data.get('performance_history', [])
        self.draw_performance_history()
    
    def update_table(self, table, rules):
        """Update a table with rules."""
        # Clear existing items
        for item in table.get_children():
            table.delete(item)
        
        # Add new items
        for rule in rules:
            table.insert("", tk.END, values=(
                rule.name,
                str(rule.condition),
                str(rule.action),
                rule.priority
            ))
    
    def draw_performance_history(self):
        """Draw the performance history graph."""
        self.performance_canvas.delete("all")
        
        if not self.performance_history:
            return
        
        # Get canvas dimensions
        width = self.performance_canvas.winfo_width()
        height = self.performance_canvas.winfo_height()
        
        # Calculate scales
        x_scale = width / max(1, len(self.performance_history) - 1)
        y_scale = height
        
        # Draw axes
        self.performance_canvas.create_line(0, height, width, height)  # X-axis
        self.performance_canvas.create_line(0, 0, 0, height)  # Y-axis
        
        # Draw performance line
        points = []
        for i, value in enumerate(self.performance_history):
            x = i * x_scale
            y = height - (value * y_scale)
            points.extend([x, y])
        
        if len(points) >= 4:  # Need at least 2 points to draw a line
            self.performance_canvas.create_line(*points, fill="blue", width=2)
    
    def on_continue(self):
        """Handle continue button click."""
        # This method should be overridden by the application
        pass
    
    def new_scenario(self):
        """Create a new scenario."""
        self.current_data = self.scenario_manager.create_new_scenario()
        self.update_interface(self.current_data)
    
    def open_scenario(self):
        """Open an existing scenario."""
        file_path = filedialog.askopenfilename(
            initialdir=self.scenario_manager.scenarios_dir,
            filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            try:
                scenario_name = os.path.basename(file_path)
                data = self.scenario_manager.load_scenario(scenario_name)
                self.update_interface(data)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load scenario: {str(e)}")
    
    def save_scenario(self):
        """Save the current scenario."""
        if not self.current_data:
            messagebox.showwarning("Warning", "No scenario to save")
            return
        
        if not self.scenario_manager.current_scenario:
            self.save_scenario_as()
        else:
            try:
                self.scenario_manager.save_scenario(
                    self.scenario_manager.current_scenario,
                    self.current_data
                )
                messagebox.showinfo("Success", "Scenario saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save scenario: {str(e)}")
    
    def save_scenario_as(self):
        """Save the current scenario with a new name."""
        if not self.current_data:
            messagebox.showwarning("Warning", "No scenario to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            initialdir=self.scenario_manager.scenarios_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            try:
                scenario_name = os.path.basename(file_path)
                self.scenario_manager.save_scenario(scenario_name, self.current_data)
                messagebox.showinfo("Success", "Scenario saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save scenario: {str(e)}")
    
    def load_demo_scenario(self, scenario_name):
        """Load a demo scenario."""
        if scenario_name in DEMO_SCENARIOS:
            self.current_data = DEMO_SCENARIOS[scenario_name]
            self.update_interface(self.current_data)
        else:
            messagebox.showerror("Error", f"Demo scenario '{scenario_name}' not found")
    
    def run(self):
        """Start the interface."""
        self.root.mainloop()
    
    def close(self):
        """Close the interface."""
        self.root.quit()
        self.root.destroy() 