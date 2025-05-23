"""
Desktop GUI interface for the Coach-Driven Search application.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import sys
import os
from pathlib import Path

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.search import SearchEngine
from core.coach import Coach
from utils.config import load_config
from utils.logger import setup_logger

class SearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Coach-Driven Search")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.config = load_config()
        self.search_engine = SearchEngine(self.config)
        self.coach = Coach(self.config)
        self.logger = setup_logger()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Style configuration
        style = ttk.Style()
        style.configure('TFrame', background='#f3f4f6')
        style.configure('Header.TLabel', 
                       font=('Helvetica', 24, 'bold'),
                       foreground='#2563eb',
                       background='#f3f4f6')
        style.configure('Section.TLabel',
                       font=('Helvetica', 12, 'bold'),
                       background='white')
        
        # Create main container with three columns
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Configure main frame columns
        main_frame.columnconfigure(0, weight=1)  # Learner's area
        main_frame.columnconfigure(1, weight=2)  # Search area
        main_frame.columnconfigure(2, weight=1)  # Coach's area
        
        # Header
        header = ttk.Label(main_frame, 
                          text="Coach-Driven Search",
                          style='Header.TLabel')
        header.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Learner's Rule Base Area
        learner_frame = ttk.LabelFrame(main_frame, text="Learner's Rule Base", padding="10")
        learner_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        learner_frame.columnconfigure(0, weight=1)
        learner_frame.rowconfigure(1, weight=1)
        
        self.learner_rules = scrolledtext.ScrolledText(learner_frame,
                                                     height=20,
                                                     wrap=tk.WORD,
                                                     font=('Helvetica', 10))
        self.learner_rules.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.learner_rules.insert('1.0', "Learner's rules will appear here...")
        self.learner_rules.configure(state='disabled')
        
        # Search Area
        search_frame = ttk.LabelFrame(main_frame, text="Search Context and Results", padding="10")
        search_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        search_frame.columnconfigure(0, weight=1)
        search_frame.rowconfigure(1, weight=1)
        search_frame.rowconfigure(3, weight=1)
        
        # Search input
        search_input_frame = ttk.Frame(search_frame)
        search_input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        search_input_frame.columnconfigure(0, weight=1)
        
        self.search_entry = ttk.Entry(search_input_frame, font=('Helvetica', 12))
        self.search_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.search_entry.bind('<Return>', lambda e: self.perform_search())
        
        search_button = ttk.Button(search_input_frame, 
                                 text="Search",
                                 command=self.perform_search)
        search_button.grid(row=0, column=1)
        
        # Context display
        context_frame = ttk.LabelFrame(search_frame, text="Current Context", padding="5")
        context_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        context_frame.columnconfigure(0, weight=1)
        context_frame.rowconfigure(0, weight=1)
        
        self.context_text = scrolledtext.ScrolledText(context_frame,
                                                    height=5,
                                                    wrap=tk.WORD,
                                                    font=('Helvetica', 10))
        self.context_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.context_text.insert('1.0', "Start State: \nGoal State: ")
        self.context_text.configure(state='disabled')
        
        # Search results
        results_frame = ttk.LabelFrame(search_frame, text="Search Paths", padding="5")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame,
                                                    height=10,
                                                    wrap=tk.WORD,
                                                    font=('Helvetica', 10))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.results_text.configure(state='disabled')
        
        # Coach's Area
        coach_frame = ttk.LabelFrame(main_frame, text="Coach's Interface", padding="10")
        coach_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        coach_frame.columnconfigure(0, weight=1)
        coach_frame.rowconfigure(1, weight=1)
        coach_frame.rowconfigure(3, weight=1)
        
        # Coach's rule base
        coach_rules_frame = ttk.LabelFrame(coach_frame, text="Coach's Rule Base", padding="5")
        coach_rules_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        coach_rules_frame.columnconfigure(0, weight=1)
        coach_rules_frame.rowconfigure(0, weight=1)
        
        self.coach_rules = scrolledtext.ScrolledText(coach_rules_frame,
                                                   height=8,
                                                   wrap=tk.WORD,
                                                   font=('Helvetica', 10))
        self.coach_rules.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.coach_rules.insert('1.0', "Coach's rules will appear here...")
        self.coach_rules.configure(state='disabled')
        
        # Coach's advice
        advice_frame = ttk.LabelFrame(coach_frame, text="Coach's Advice", padding="5")
        advice_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        advice_frame.columnconfigure(0, weight=1)
        advice_frame.rowconfigure(0, weight=1)
        
        self.advice_text = scrolledtext.ScrolledText(advice_frame,
                                                   height=8,
                                                   wrap=tk.WORD,
                                                   font=('Helvetica', 10))
        self.advice_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.advice_text.configure(state='disabled')
        
        # Configure all frames to expand with window
        for frame in (learner_frame, search_frame, coach_frame):
            frame.rowconfigure(0, weight=1)
    
    def perform_search(self):
        """Execute the search and update the UI."""
        query = self.search_entry.get().strip()
        if not query:
            return
            
        # Get coach's analysis
        analysis = self.coach.analyze_query(query)
        
        # Get search results
        results = self.search_engine.search(query)
        
        # Get improvement suggestions
        suggestions = self.coach.suggest_improvements(query)
        
        # Update UI
        self.update_text_widget(self.context_text, f"Start State: {query}\nGoal State: {query} (modified)")
        
        results_text = ""
        for result in results['results']:
            results_text += f"Path: {result['title']}\nURL: {result['url']}\n\n"
        self.update_text_widget(self.results_text, results_text)
        
        self.update_text_widget(self.advice_text, analysis + "\n\n" + "\n".join(f"• {suggestion}" for suggestion in suggestions))
    
    def update_text_widget(self, widget, text):
        """Update a text widget with new content."""
        widget.configure(state='normal')
        widget.delete('1.0', tk.END)
        widget.insert('1.0', text)
        widget.configure(state='disabled')

def main():
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 