# layout.py
import tkinter as tk
from tkinter import ttk
from display import PlaceholderEntry

class SpecificationPanel(tk.LabelFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, text="  Specification  ", padx=5, pady=5, **kwargs)
        # Create custom style using the provided master.
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("Custom.Treeview",
                        background="white",
                        fieldbackground="white",
                        bordercolor="black",
                        borderwidth=1,
                        relief="solid")
        self.create_widgets()

    def create_widgets(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        state_frame = tk.Frame(self)
        state_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        state_label = tk.Label(state_frame, text="State Space:", anchor="w", width=9)
        state_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.state_space_entry = PlaceholderEntry(state_frame,
            placeholder="e.g., Color: {red, blue, green}; Size: {small, medium, large}; Shape: {circle, square}",
            width=40)
        self.state_space_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        state_frame.grid_columnconfigure(1, weight=1, minsize=80)
        
        tree_frame = tk.Frame(self)
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        columns = ("Priority", "Rule Name", "Condition", "Preference")
        self.rules_tree = ttk.Treeview(tree_frame, columns=columns, style="Custom.Treeview", show="headings")
        for col in columns:
            self.rules_tree.heading(col, text=col)
        self.rules_tree.column("Priority", width=80, anchor="w")
        self.rules_tree.column("Rule Name", width=120, anchor="w")
        self.rules_tree.column("Condition", width=250, anchor="w")
        self.rules_tree.column("Preference", width=120, anchor="w")
        self.rules_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(tree_frame, orient="vertical", command=self.rules_tree.yview)
        scrollbar.config(width=15)
        self.rules_tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

class ReasoningPanel(tk.LabelFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, text="  Reasoning  ", padx=5, pady=5, **kwargs)
        self.create_widgets()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

    def create_widgets(self):
        container = tk.Frame(self)
        container.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        container.grid_columnconfigure(0, weight=0)
        container.grid_columnconfigure(1, weight=1, minsize=80)
        container.grid_columnconfigure(2, weight=0)

        initial_label = tk.Label(container, text="Initial State:", anchor="w", width=9)
        initial_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.initial_state_entry = PlaceholderEntry(container,
            placeholder="e.g., Color=red; Size=small; Shape=circle", width=40, justify="left")
        self.initial_state_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.search_button = tk.Button(container, text="Search", width=8)
        self.search_button.grid(row=0, column=2, rowspan=2, padx=5, pady=2, sticky="nsew")

        target_label = tk.Label(container, text="Target State:", anchor="w", width=9)
        target_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.target_state_entry = PlaceholderEntry(container,
            placeholder="e.g., Size=large (partial allowed)", width=40, justify="left")
        self.target_state_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        self.trace_slider = tk.Scale(container, from_=0, to=0, orient="horizontal",
                                     resolution=1, showvalue=False, sliderlength=30)
        self.trace_slider.grid(row=2, column=0, columnspan=2, padx=5, pady=(5,2), sticky="ew")
        nav_frame = tk.Frame(container)
        nav_frame.grid(row=2, column=2, padx=5, pady=2, sticky="ew")
        nav_frame.grid_columnconfigure(0, weight=1)
        nav_frame.grid_columnconfigure(1, weight=1)
        self.prev_button = tk.Button(nav_frame, text="<", width=4)
        self.prev_button.grid(row=0, column=0, sticky="ew")
        self.next_button = tk.Button(nav_frame, text=">", width=4)
        self.next_button.grid(row=0, column=1, sticky="ew")
        
        result_frame = tk.Frame(self)
        result_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_rowconfigure(0, weight=1)
        columns = ("Current State", "Transition")
        self.result_tree = ttk.Treeview(result_frame, columns=columns, style="Custom.Treeview", show="headings")
        for col in columns:
            self.result_tree.heading(col, text=col)
        self.result_tree.column("Current State", width=600, anchor="w")
        self.result_tree.column("Transition", width=150, anchor="w")
        self.result_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(result_frame, orient="vertical", command=self.result_tree.yview)
        scrollbar.config(width=15)
        self.result_tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")
