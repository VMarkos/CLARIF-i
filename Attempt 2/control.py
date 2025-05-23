# control.py
import json
from tkinter import filedialog, messagebox
from engines import parse_assignment, parse_variable_domains, dfs_collect_traces

class ControlMixin:
    def file_new(self):
        # Clear the state space entry and re-add its placeholder text (greyed-out)
        self.spec_panel.state_space_entry.delete(0, "end")
        self.spec_panel.state_space_entry._add_placeholder()
        # Clear the initial state entry and re-add its placeholder text
        self.reasoning_panel.initial_state_entry.delete(0, "end")
        self.reasoning_panel.initial_state_entry._add_placeholder()
        # Clear the target state entry and re-add its placeholder text
        self.reasoning_panel.target_state_entry.delete(0, "end")
        self.reasoning_panel.target_state_entry._add_placeholder()
        for item in self.spec_panel.rules_tree.get_children():
            self.spec_panel.rules_tree.delete(item)
        for item in self.reasoning_panel.result_tree.get_children():
            self.reasoning_panel.result_tree.delete(item)
        self.current_filename = None
        self.next_rule_priority = 1
        self.trace = []
        self.traces = []
        self.status_table = {}
        self.prev_state_space = self.spec_panel.state_space_entry.get()

    def file_save_as(self):
        data = {
            "state_space": self.spec_panel.state_space_entry.get(),
            "initial_state": self.reasoning_panel.initial_state_entry.get(),
            "target_state": self.reasoning_panel.target_state_entry.get(),
            "rules": [self.spec_panel.rules_tree.item(item)["values"] for item in self.spec_panel.rules_tree.get_children()]
        }
        filename = filedialog.asksaveasfilename(defaultextension=".json",
                                                filetypes=[("JSON files", "*.json")])
        if filename:
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            self.current_filename = filename

    def file_save(self):
        if self.current_filename:
            data = {
                "state_space": self.spec_panel.state_space_entry.get(),
                "initial_state": self.reasoning_panel.initial_state_entry.get(),
                "target_state": self.reasoning_panel.target_state_entry.get(),
                "rules": [self.spec_panel.rules_tree.item(item)["values"] for item in self.spec_panel.rules_tree.get_children()]
            }
            with open(self.current_filename, "w") as f:
                json.dump(data, f, indent=2)
        else:
            self.file_save_as()

    def file_load(self):
        filename = filedialog.askopenfilename(defaultextension=".json",
                                              filetypes=[("JSON files", "*.json")])
        if filename:
            with open(filename, "r") as f:
                data = json.load(f)
            self.spec_panel.state_space_entry.delete(0, "end")
            self.spec_panel.state_space_entry.insert(0, data.get("state_space", ""))
            self.reasoning_panel.initial_state_entry.delete(0, "end")
            self.reasoning_panel.initial_state_entry.insert(0, data.get("initial_state", ""))
            self.reasoning_panel.target_state_entry.delete(0, "end")
            self.reasoning_panel.target_state_entry.insert(0, data.get("target_state", ""))
            for item in self.spec_panel.rules_tree.get_children():
                self.spec_panel.rules_tree.delete(item)
            self.next_rule_priority = 1
            for rule in data.get("rules", []):
                if any(str(val).strip() for val in rule):
                    self.spec_panel.rules_tree.insert("", "end", values=rule)
                    try:
                        num = int(rule[0])
                        if num >= self.next_rule_priority:
                            self.next_rule_priority = num + 1
                    except Exception:
                        pass
            self.current_filename = filename
            self.prev_state_space = self.spec_panel.state_space_entry.get()
            for item in self.reasoning_panel.result_tree.get_children():
                self.reasoning_panel.result_tree.delete(item)
            self.trace = []
            self.traces = []
            self.status_table = {}

    def file_exit(self):
        self.quit()

    def load_demo_scenario1(self):
        self.clear_outputs_on_update(None)
        demo_state_space = "Color: {red, blue, green}; Size: {small, medium, large}; Shape: {circle, square}"
        self.spec_panel.state_space_entry.delete(0, "end")
        self.spec_panel.state_space_entry.insert(0, demo_state_space)
        self.spec_panel.state_space_entry.config(fg="black")
        for item in self.spec_panel.rules_tree.get_children():
            self.spec_panel.rules_tree.delete(item)
        self.next_rule_priority = 1
        self.add_rule_row("R1", "Color=red", "Size=large")
        self.add_rule_row("R2", "Size=small", "Color=blue")
        self.add_rule_row("R3", "Color=red", "Size=medium")
        self.reasoning_panel.initial_state_entry.delete(0, "end")
        self.reasoning_panel.initial_state_entry.insert(0, "Color=red; Size=small; Shape=circle")
        self.reasoning_panel.initial_state_entry.config(fg="black")
        self.reasoning_panel.target_state_entry.delete(0, "end")
        self.reasoning_panel.target_state_entry.insert(0, "Size=large")
        self.reasoning_panel.target_state_entry.config(fg="black")
        self.clear_outputs_on_update(None)

    def load_demo_scenario2(self):
        self.clear_outputs_on_update(None)
        demo_state_space = "Weather: {sunny, cloudy, rainy}; Temperature: {cold, mild, hot}; Time: {morning, afternoon, evening}"
        self.spec_panel.state_space_entry.delete(0, "end")
        self.spec_panel.state_space_entry.insert(0, demo_state_space)
        self.spec_panel.state_space_entry.config(fg="black")
        for item in self.spec_panel.rules_tree.get_children():
            self.spec_panel.rules_tree.delete(item)
        self.next_rule_priority = 1
        self.add_rule_row("R4", "Weather=sunny", "Temperature=hot")
        self.add_rule_row("R5", "Temperature=cold", "Weather=rainy")
        self.add_rule_row("R6", "Time=morning", "Weather=sunny")
        self.add_rule_row("R7", "Weather=rainy", "Temperature=mild")
        self.reasoning_panel.initial_state_entry.delete(0, "end")
        self.reasoning_panel.initial_state_entry.insert(0, "Weather=cloudy; Temperature=cold; Time=morning")
        self.reasoning_panel.initial_state_entry.config(fg="black")
        self.reasoning_panel.target_state_entry.delete(0, "end")
        self.reasoning_panel.target_state_entry.insert(0, "Temperature=hot")
        self.reasoning_panel.target_state_entry.config(fg="black")
        self.clear_outputs_on_update(None)

    def add_rule_row(self, rule_name, condition, preference):
        self.spec_panel.rules_tree.insert("", "end", values=(str(self.next_rule_priority), rule_name, condition, preference))
        self.next_rule_priority += 1

    def clear_outputs_on_update(self, event):
        # Clear the output tree completely.
        for item in self.reasoning_panel.result_tree.get_children():
            self.reasoning_panel.result_tree.delete(item)
        # Clear selection and tags in the rules tree.
        self.spec_panel.rules_tree.selection_clear()
        for item in self.spec_panel.rules_tree.get_children():
            self.spec_panel.rules_tree.item(item, tags=[])
        # Reset stored trace data and status table.
        self.trace = []
        self.traces = []
        self.status_table = {}

    def get_rules(self):
        rules = []
        for idx, item in enumerate(self.spec_panel.rules_tree.get_children()):
            prio_text = self.spec_panel.rules_tree.set(item, "Priority")
            if prio_text.strip() == "":
                continue
            try:
                prio = int(prio_text)
            except ValueError:
                prio = 0
            rule_name = self.spec_panel.rules_tree.set(item, "Rule Name")
            condition_text = self.spec_panel.rules_tree.set(item, "Condition")
            preference_text = self.spec_panel.rules_tree.set(item, "Preference")
            if not rule_name and not condition_text and not preference_text:
                continue
            condition = parse_assignment(condition_text)
            preference = parse_assignment(preference_text)
            if len(preference) != 1:
                messagebox.showerror("Rule Error", f"Rule '{rule_name}' must assign exactly one variable in the preference.")
                return None
            rules.append({
                "rule_name": rule_name if rule_name else f"Rule {len(rules)+1}",
                "condition": condition,
                "preference": preference,
                "priority": prio
            })
        return rules
