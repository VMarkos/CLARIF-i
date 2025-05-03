# manager.py
import tkinter as tk
from layout import SpecificationPanel, ReasoningPanel
from engines import parse_assignment, dfs_collect_traces, compute_statuses
from control import ControlMixin
from events import EventMixin

class PreferenceReasoningApp(ControlMixin, EventMixin, tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Conditional Preference Reasoning System")
        self.geometry("900x700")
        # Storage for traces and search state.
        self.trace = []         # Current displayed trace: list of (state, transition) pairs
        self.traces = []        # All collected traces (each a tuple (trace, score))
        self.status_table = {}  # (Not used by the new highlighting, but kept if needed)
        self.next_rule_priority = 1
        self.current_trace_index = 0
        self.current_filename = None
        self.prev_state_space = self.get_default_state_space()
        self.create_menu()
        self.create_widgets()
        self.bind_events()

    def get_default_state_space(self):
        return "e.g., Color: {red, blue, green}; Size: {small, medium, large}; Shape: {circle, square}"

    def create_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self.file_new)
        file_menu.add_command(label="Save", command=self.file_save)
        file_menu.add_command(label="Save As", command=self.file_save_as)
        file_menu.add_command(label="Load", command=self.file_load)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.file_exit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        demo_menu = tk.Menu(menubar, tearoff=0)
        demo_menu.add_command(label="Demo Scenario 1", command=self.load_demo_scenario1)
        demo_menu.add_command(label="Demo Scenario 2", command=self.load_demo_scenario2)
        demo_menu.add_command(label="Demo Scenario 3", command=self.load_demo_scenario3)
        menubar.add_cascade(label="Demo", menu=demo_menu)
        
        self.config(menu=menubar)

    def create_widgets(self):
        # Create a container grid to hold both panels.
        container = tk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)
        
        self.spec_panel = SpecificationPanel(container)
        self.spec_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.reasoning_panel = ReasoningPanel(container)
        self.reasoning_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
    def bind_events(self):
        self.spec_panel.rules_tree.bind("<ButtonRelease-1>", self.on_rule_tree_select)
        self.reasoning_panel.result_tree.bind("<ButtonRelease-1>", self.on_result_tree_select)

    def on_search(self):
        init_text = self.reasoning_panel.initial_state_entry.get()
        target_text = self.reasoning_panel.target_state_entry.get()
        initial_state = parse_assignment(init_text)
        target_state = parse_assignment(target_text)
        rules = []
        for child in self.spec_panel.rules_tree.get_children():
            values = self.spec_panel.rules_tree.item(child, "values")
            if values:
                try:
                    priority = int(values[0])
                except:
                    priority = 0
                rule = {
                    "rule_name": values[1],
                    "condition": parse_assignment(values[2]),
                    "preference": parse_assignment(values[3]),
                    "priority": priority
                }
                rules.append(rule)
        if not rules:
            rules.append({
                "rule_name": "dummy",
                "condition": {},
                "preference": {"Color": "red"},
                "priority": 1
            })
        visited = set()
        collected = []
        dfs_collect_traces(initial_state, target_state, rules, visited, [], collected)
        if not collected:
            collected.append(([(initial_state, "No move")], 0))
        self.traces = collected
        self.current_trace_index = 0
        self.update_trace_display()

    def update_trace_display(self):
        for item in self.reasoning_panel.result_tree.get_children():
            self.reasoning_panel.result_tree.delete(item)
        current_trace, score = self.traces[self.current_trace_index]
        for state, transition in current_trace:
            state_str = "; ".join(f"{k}={v}" for k, v in state.items())
            self.reasoning_panel.result_tree.insert("", tk.END, values=(state_str, transition))

    def get_rules(self):
        rules = []
        for child in self.spec_panel.rules_tree.get_children():
            values = self.spec_panel.rules_tree.item(child, "values")
            if values:
                try:
                    priority = int(values[0])
                except:
                    priority = 0
                rule = {
                    "rule_name": values[1],
                    "condition": parse_assignment(values[2]),
                    "preference": parse_assignment(values[3]),
                    "priority": priority
                }
                rules.append(rule)
        return rules

    # Dummy file and demo methods.
    def file_new(self):
        self.spec_panel.state_space_entry.delete(0, tk.END)
        self.spec_panel.state_space_entry.insert(0, self.spec_panel.state_space_entry.placeholder)
        self.reasoning_panel.initial_state_entry.delete(0, tk.END)
        self.reasoning_panel.initial_state_entry.insert(0, self.reasoning_panel.initial_state_entry.placeholder)
        self.reasoning_panel.target_state_entry.delete(0, tk.END)
        self.reasoning_panel.target_state_entry.insert(0, self.reasoning_panel.target_state_entry.placeholder)
        for child in self.spec_panel.rules_tree.get_children():
            self.spec_panel.rules_tree.delete(child)
        for child in self.reasoning_panel.result_tree.get_children():
            self.reasoning_panel.result_tree.delete(child)
        self.traces = []
        self.current_trace_index = 0

    def file_save(self):
        pass

    def file_save_as(self):
        pass

    def file_load(self):
        pass

    def file_exit(self):
        self.destroy()

    def load_demo_scenario1(self):
        self.file_new()
        self.spec_panel.state_space_entry.delete(0, tk.END)
        self.spec_panel.state_space_entry.insert(0, "Color: {red, blue, green}; Size: {small, medium, large}; Shape: {circle, square}")
        self.spec_panel.rules_tree.insert("", tk.END, values=("1", "R1", "Color=red", "Size=medium"))
        self.spec_panel.rules_tree.insert("", tk.END, values=("2", "R2", "Size=medium", "Color=green"))
        self.spec_panel.rules_tree.insert("", tk.END, values=("3", "R3", "Color=green", "Size=large"))
        self.spec_panel.rules_tree.insert("", tk.END, values=("4", "R4", "Shape=circle", "Color=red"))
        self.spec_panel.rules_tree.insert("", tk.END, values=("5", "R5", "Color=blue", "Shape=square"))
        self.reasoning_panel.initial_state_entry.delete(0, tk.END)
        self.reasoning_panel.initial_state_entry.insert(0, "Color=red; Size=small; Shape=circle")
        self.reasoning_panel.target_state_entry.delete(0, tk.END)
        self.reasoning_panel.target_state_entry.insert(0, "Size=large")

    def load_demo_scenario2(self):
        self.file_new()
        self.spec_panel.state_space_entry.delete(0, tk.END)
        self.spec_panel.state_space_entry.insert(0, "Weather: {sunny, cloudy, rainy}; Temperature: {cold, mild, hot}; Day: {weekday, weekend}")
        self.spec_panel.rules_tree.insert("", tk.END, values=("1", "R1", "Weather=sunny", "Temperature=hot"))
        self.spec_panel.rules_tree.insert("", tk.END, values=("2", "R2", "Day=weekday", "Weather=rainy"))
        self.spec_panel.rules_tree.insert("", tk.END, values=("3", "R3", "Temperature=cold", "Weather=cloudy"))
        self.spec_panel.rules_tree.insert("", tk.END, values=("4", "R4", "Weather=rainy", "Temperature=mild"))
        self.spec_panel.rules_tree.insert("", tk.END, values=("5", "R5", "Temperature=mild", "Weather=sunny"))
        self.reasoning_panel.initial_state_entry.delete(0, tk.END)
        self.reasoning_panel.initial_state_entry.insert(0, "Weather=cloudy; Temperature=cold; Day=weekday")
        self.reasoning_panel.target_state_entry.delete(0, tk.END)
        self.reasoning_panel.target_state_entry.insert(0, "Temperature=hot")

    def load_demo_scenario3(self):
        self.file_new()
        self.spec_panel.state_space_entry.delete(0, tk.END)
        self.spec_panel.state_space_entry.insert(0, "Speed: {slow, moderate, fast}; Fuel: {low, medium, high}; Engine: {off, on}")
        self.spec_panel.rules_tree.insert("", tk.END, values=("1", "R1", "Fuel=high", "Engine=on"))
        self.spec_panel.rules_tree.insert("", tk.END, values=("2", "R2", "Engine=on", "Speed=moderate"))
        self.spec_panel.rules_tree.insert("", tk.END, values=("3", "R3", "Speed=moderate", "Fuel=medium"))
        self.spec_panel.rules_tree.insert("", tk.END, values=("4", "R4", "Fuel=medium", "Speed=fast"))
        self.spec_panel.rules_tree.insert("", tk.END, values=("5", "R5", "Engine=off", "Fuel=high"))
        self.reasoning_panel.initial_state_entry.delete(0, tk.END)
        self.reasoning_panel.initial_state_entry.insert(0, "Speed=slow; Fuel=high; Engine=off")
        self.reasoning_panel.target_state_entry.delete(0, tk.END)
        self.reasoning_panel.target_state_entry.insert(0, "Engine=on; Speed=fast")

if __name__ == "__main__":
    app = PreferenceReasoningApp()
    app.mainloop()
