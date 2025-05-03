# events.py
import tkinter as tk
from engines import compute_statuses, parse_assignment

def parse_state_string(state_str):
    state = {}
    for pair in state_str.split(';'):
        pair = pair.strip()
        if pair and '=' in pair:
            key, value = pair.split('=', 1)
            state[key.strip()] = value.strip()
    return state

def rule_triggered(rule, state):
    condition = rule.get("condition", {})
    for key, expected in condition.items():
        if state.get(key) != expected:
            return False
    return True

class EventMixin:
    TAG_TRIGGERED = {"background": "lightgreen", "foreground": "black"}
    TAG_NOT_TRIGGERED = {"background": "white", "foreground": "black"}
    TAG_CLICK = {"background": "orange", "foreground": "black"}
    
    def reset_highlighting(self):
        for item in self.spec_panel.rules_tree.get_children():
            self.spec_panel.rules_tree.item(item, tags=())
        for item in self.reasoning_panel.result_tree.get_children():
            self.reasoning_panel.result_tree.item(item, tags=())
    
    def on_rule_tree_select(self, event):
        try:
            rule_tree = self.spec_panel.rules_tree
            rule_id = rule_tree.identify_row(event.y) if event else (rule_tree.selection()[0] if rule_tree.selection() else None)
            if not rule_id:
                return
            rule_tree.selection_set(rule_id)
            self.reasoning_panel.result_tree.selection_clear()
            self.reset_highlighting()
            values = rule_tree.item(rule_id, "values")
            if not values or len(values) < 4:
                return
            selected_rule = {
                "rule_name": values[1],
                "condition": parse_assignment(values[2]),
                "preference": parse_assignment(values[3])
            }
            rule_tree.item(rule_id, tags=("click",))
            rule_tree.tag_configure("click", **self.TAG_CLICK)
            
            res_tree = self.reasoning_panel.result_tree
            for idx, state_id in enumerate(res_tree.get_children()):
                state_str = res_tree.item(state_id, "values")[0]
                state = parse_state_string(state_str)
                tag = "triggered" if rule_triggered(selected_rule, state) else "not_triggered"
                res_tree.item(state_id, tags=(tag,))
            res_tree.tag_configure("triggered", **self.TAG_TRIGGERED)
            res_tree.tag_configure("not_triggered", **self.TAG_NOT_TRIGGERED)
        except Exception as e:
            print("Error in on_rule_tree_select:", e)
    
    def on_result_tree_select(self, event):
        try:
            res_tree = self.reasoning_panel.result_tree
            state_id = res_tree.identify_row(event.y) if event else (res_tree.selection()[0] if res_tree.selection() else None)
            if not state_id:
                return
            res_tree.selection_set(state_id)
            self.spec_panel.rules_tree.selection_clear()
            self.reset_highlighting()
            state_str = res_tree.item(state_id, "values")[0]
            selected_state = parse_state_string(state_str)
            res_tree.item(state_id, tags=("click",))
            res_tree.tag_configure("click", **self.TAG_CLICK)
            rule_tree = self.spec_panel.rules_tree
            for rule_id in rule_tree.get_children():
                values = rule_tree.item(rule_id, "values")
                if not values or len(values) < 4:
                    continue
                rule = {
                    "rule_name": values[1],
                    "condition": parse_assignment(values[2]),
                    "preference": parse_assignment(values[3])
                }
                tag = "triggered" if rule_triggered(rule, selected_state) else "not_triggered"
                rule_tree.item(rule_id, tags=(tag,))
            rule_tree.tag_configure("triggered", **self.TAG_TRIGGERED)
            rule_tree.tag_configure("not_triggered", **self.TAG_NOT_TRIGGERED)
        except Exception as e:
            print("Error in on_result_tree_select:", e)
