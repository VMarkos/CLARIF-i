#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog, scrolledtext
import math

##########################################################################
#                           Model Section                                #
##########################################################################

class CPRule:
    def __init__(self, condition, ordering):
        """
        condition: Dictionary mapping variable names to fixed values (as strings), e.g. {'B': "1"} or {'Dinner': "meat"}.
        ordering: List of two strings; e.g. ["1", "0"] means for the head variable the first value is preferred over the second.
        """
        self.condition = condition
        self.ordering = ordering

    def applies(self, outcome):
        for var, val in self.condition.items():
            if outcome.get(var) != val:
                return False
        return True

    def __repr__(self):
        return f"If {self.condition} then {self.ordering}"


class CPTable:
    def __init__(self, variable):
        self.variable = variable   # The variable for which this table applies.
        self.rules = []            # List of CPRule objects (initially empty).

    def add_rule(self, condition, ordering):
        self.rules.append(CPRule(condition, ordering))

    def get_dependencies(self):
        deps = set()
        for rule in self.rules:
            deps.update(rule.condition.keys())
        return deps

    def get_preference(self, outcome):
        """
        For a given outcome (a dict mapping variables to current values as strings),
        returns (preferred_value, explanation) using the last applicable rule.
        """
        applied_rule = None
        for rule in self.rules:
            if rule.applies(outcome):
                applied_rule = rule
        if applied_rule:
            explanation = f"Rule: if {applied_rule.condition} then {applied_rule.ordering}."
            idx = self.rules.index(applied_rule)
            for higher in self.rules[idx+1:]:
                if not higher.applies(outcome):
                    explanation += f" (Higher rule {higher.condition} inactive)"
            return applied_rule.ordering[0], explanation
        else:
            return outcome[self.variable], "(No rule)"
    
    def __repr__(self):
        deps = self.get_dependencies()
        deps_str = ", ".join(deps) if deps else "none"
        s = f"{self.variable}: [Depends on: {deps_str}]\n"
        for rule in self.rules:
            s += f"  if {rule.condition} then {rule.ordering}\n"
        return s


class CPNet:
    def __init__(self, variables):
        """
        variables: A list of variable names.
        """
        self.variables = variables[:]  # Make a copy.
        self.cpt = {var: CPTable(var) for var in variables}

    def __repr__(self):
        s = ""
        for var in self.variables:
            s += repr(self.cpt[var]) + "\n"
        return s

##########################################################################
#                      Reasoning and XML Functions                       #
##########################################################################

def hamming_distance(t1, t2):
    return sum(1 for a, b in zip(t1, t2) if a != b)

def outcome_to_tuple(outcome, variables):
    """Converts an outcome dict into a tuple in the order of variables."""
    return tuple(outcome[var] for var in variables)

def parse_outcome(binary_str, variables):
    """
    Parses a binary outcome string (e.g. "101") into a dictionary mapping each variable to its value (as a string).
    """
    s = binary_str.strip()
    if len(s) != len(variables):
        raise ValueError(f"Outcome must have exactly {len(variables)} digits.")
    outcome = {}
    for i, ch in enumerate(s):
        if ch not in "01":
            raise ValueError("Outcome must consist only of 0s and 1s.")
        outcome[variables[i]] = ch
    return outcome

def parse_conditions(cond_str):
    """
    Parses a string like "B=1, D=0" into a dictionary.
    An empty string yields {}.
    """
    cond_str = cond_str.strip()
    if not cond_str:
        return {}
    cond = {}
    for part in cond_str.split(','):
        if "=" in part:
            var, val = part.split("=")
            var = var.strip()
            val = val.strip()
            if val:
                cond[var] = val
    return cond

def is_flip_justified(cp_table, outcome):
    """
    For a given CP-table and current outcome, a worsening flip is justified if and only if:
      (a) There is an applicable rule whose conclusion equals the current (preferred) value.
      (b) And no higher-priority applicable rule favors the inverse value.
    Returns (True, explanation) if justified; otherwise (False, explanation).
    """
    applied_rule = None
    for rule in cp_table.rules:
        if rule.applies(outcome):
            applied_rule = rule
    if not applied_rule:
        return False, "No applicable rule."
    preferred = applied_rule.ordering[0]
    current_val = outcome.get(cp_table.variable)
    if current_val != preferred:
        return False, f"Current value {current_val} ≠ preferred {preferred}."
    idx = cp_table.rules.index(applied_rule)
    inverse = "1" if preferred == "0" else "0"
    for higher_rule in cp_table.rules[idx+1:]:
        if higher_rule.applies(outcome) and (higher_rule.ordering[0] == inverse):
            return False, f"Higher rule {higher_rule.condition} favors {inverse}."
    return True, f"Justified by rule {applied_rule.condition}."

def find_worsening_flipping_sequence(cpnet, source, target):
    """
    Starting from the source outcome (more preferred), performs a BFS over outcomes by applying worsening flips.
    A flip for a variable is allowed only if is_flip_justified(cpnet.cpt[var], outcome) returns True.
    Returns (sequence, exact) where sequence is a list of tuples:
        (flipped_variable, justification, new outcome tuple)
    and exact is True if the target was reached.
    (Assumes binary domains.)
    """
    variables = cpnet.variables
    src = outcome_to_tuple(source, variables)
    tgt = outcome_to_tuple(target, variables)
    visited = {src: (None, None, None)}
    queue = deque([src])
    best, best_d = src, hamming_distance(src, tgt)
    found = False
    while queue:
        current = queue.popleft()
        if current == tgt:
            found = True
            break
        curr_outcome = {var: val for var, val in zip(variables, current)}
        for i, var in enumerate(variables):
            justified, justification = is_flip_justified(cpnet.cpt[var], curr_outcome)
            if justified:
                new_outcome = list(current)
                new_outcome[i] = "1" if curr_outcome[var] != "1" else "0"
                new_tuple = tuple(new_outcome)
                if new_tuple not in visited:
                    visited[new_tuple] = (current, var, justification)
                    queue.append(new_tuple)
                    d = hamming_distance(new_tuple, tgt)
                    if d < best_d:
                        best, best_d = new_tuple, d
    seq = []
    cur = tgt if found else best
    while cur != src:
        parent, var, justification = visited[cur]
        seq.append((var, justification, cur))
        cur = parent
    seq.reverse()
    return seq, found

def save_cpnet_to_xml(cpnet, filename):
    root = ET.Element("CPNet")
    vars_elem = ET.SubElement(root, "Variables")
    for var in cpnet.variables:
        var_elem = ET.SubElement(vars_elem, "Variable")
        var_elem.text = var
    tables_elem = ET.SubElement(root, "CPTables")
    for var in cpnet.variables:
        table = cpnet.cpt[var]
        table_elem = ET.SubElement(tables_elem, "CPTable", variable=var)
        for rule in table.rules:
            rule_elem = ET.SubElement(table_elem, "Rule")
            cond_elem = ET.SubElement(rule_elem, "Condition")
            cond_elem.text = ", ".join(f"{k}={v}" for k, v in rule.condition.items())
            order_elem = ET.SubElement(rule_elem, "Ordering")
            order_elem.text = f"{rule.ordering[0]}>{rule.ordering[1]}"
    tree = ET.ElementTree(root)
    tree.write(filename, encoding="utf-8", xml_declaration=True)

def load_cpnet_from_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    vars_elem = root.find("Variables")
    variables = [v.text for v in vars_elem.findall("Variable")]
    cpnet = CPNet(variables)
    tables_elem = root.find("CPTables")
    if tables_elem is not None:
        for table_elem in tables_elem.findall("CPTable"):
            var = table_elem.attrib.get("variable")
            table = cpnet.cpt.get(var)
            if table is None:
                continue
            for rule_elem in table_elem.findall("Rule"):
                cond_elem = rule_elem.find("Condition")
                cond_str = cond_elem.text if cond_elem is not None else ""
                condition = parse_conditions(cond_str)
                order_elem = rule_elem.find("Ordering")
                order_str = order_elem.text if order_elem is not None else ""
                if ">" in order_str:
                    parts = order_str.split(">")
                    ordering = [parts[0].strip(), parts[1].strip()]
                else:
                    ordering = ["1", "0"]
                table.add_rule(condition, ordering)
    return cpnet

##########################################################################
#                           GUI Section                                  #
##########################################################################

class CPNetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ceteris Paribus Coaching")
        # Initialize with 3 variables: A, B, C.
        self.variable_count = 3
        self.cpnet = CPNet([chr(65+i) for i in range(self.variable_count)])
        # For each variable, store a domain (default binary, but editable).
        self.domains = {v: ["1", "0"] for v in self.cpnet.variables}
        self.cpnet_filename = None
        self.zoom_factor = 1.0
        self.cpt_frames = {}  # For highlighting CP-table frames.
        self.create_menu()
        self.create_widgets()

    # ---------------- Menu ----------------
    def create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self.new_cpnet)
        file_menu.add_command(label="Open...", command=self.open_cpnet)
        file_menu.add_command(label="Save", command=self.save_cpnet)
        file_menu.add_command(label="Save As...", command=self.save_cpnet_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label=" File ", menu=file_menu)
        self.root.config(menu=menubar)

    # ---------------- File Menu Functions ----------------
    def new_cpnet(self):
        num = simpledialog.askinteger("New CP-net", "Enter the number of variables:",
                                      initialvalue=3, minvalue=1, maxvalue=10)
        if num is not None:
            self.variable_count = num
            self.cpnet = CPNet([chr(65+i) for i in range(num)])
            self.domains = {v: ["1", "0"] for v in self.cpnet.variables}
            self.cpnet_filename = None
            self.src_entry.delete(0, tk.END)
            self.tgt_entry.delete(0, tk.END)
            self.update_all()

    def open_cpnet(self):
        filename = filedialog.askopenfilename(title="Open CP-net", filetypes=[("XML Files", "*.xml")])
        if filename:
            try:
                self.cpnet = load_cpnet_from_xml(filename)
                self.variable_count = len(self.cpnet.variables)
                self.domains = {v: ["1", "0"] for v in self.cpnet.variables}
                self.cpnet_filename = filename
                self.update_all()
            except Exception as e:
                messagebox.showerror("Open Error", str(e))

    def save_cpnet(self):
        if self.cpnet_filename:
            try:
                save_cpnet_to_xml(self.cpnet, self.cpnet_filename)
                messagebox.showinfo("Save", "CP-net saved successfully.")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))
        else:
            self.save_cpnet_as()

    def save_cpnet_as(self):
        filename = filedialog.asksaveasfilename(title="Save CP-net As", defaultextension=".xml",
                                                  filetypes=[("XML Files", "*.xml")])
        if filename:
            try:
                save_cpnet_to_xml(self.cpnet, filename)
                self.cpnet_filename = filename
                messagebox.showinfo("Save As", "CP-net saved successfully.")
            except Exception as e:
                messagebox.showerror("Save As Error", str(e))

    # ---------------- Parsing Functions ----------------
    def parse_outcome_gui(self, outcome_str):
        """Expects outcomes as a comma-separated list (one value per variable), matched against the domain."""
        tokens = [t.strip() for t in outcome_str.split(",")]
        if len(tokens) != len(self.cpnet.variables):
            raise ValueError(f"Expected {len(self.cpnet.variables)} values; got {len(tokens)}.")
        outcome = {}
        for i, var in enumerate(self.cpnet.variables):
            domain = self.domains.get(var, [])
            token = tokens[i]
            matches = [d for d in domain if d.lower() == token.lower()]
            if len(matches) == 1:
                outcome[var] = matches[0]
            else:
                raise ValueError(f"Value '{token}' for variable {var} not found in domain {domain}.")
        return outcome

    def match_variable(self, var_str):
        var_str = var_str.strip()
        matches = [v for v in self.cpnet.variables if v.lower() == var_str.lower()]
        return matches[0] if len(matches) == 1 else None

    def match_value(self, variable, val_str):
        val_str = val_str.strip()
        domain = self.domains.get(variable, [])
        matches = [d for d in domain if d.lower() == val_str.lower()]
        return matches[0] if len(matches) == 1 else None

    def parse_advice_input(self, advice_str):
        """
        Parses a free-form if-condition into a dictionary.
        Accepts tokens such as "B=1", "b1", or a lone value (e.g. "meat") if unambiguous.
        Returns a dict mapping variable to value.
        """
        tokens = [t.strip() for t in advice_str.replace(";", ",").split(",") if t.strip()]
        conditions = {}
        for token in tokens:
            if "=" in token:
                var_part, val_part = token.split("=", 1)
                var_candidate = self.match_variable(var_part)
                if not var_candidate:
                    raise ValueError(f"Ambiguous or unknown variable in '{var_part}'.")
                val_candidate = self.match_value(var_candidate, val_part)
                if not val_candidate:
                    raise ValueError(f"Value '{val_part}' not found in domain of {var_candidate}.")
                conditions[var_candidate] = val_candidate
            else:
                # Try to separate letters and digits.
                if token[0].isalpha():
                    i = 0
                    while i < len(token) and token[i].isalpha():
                        i += 1
                    var_part = token[:i]
                    val_part = token[i:]
                    if var_part and val_part:
                        var_candidate = self.match_variable(var_part)
                        if var_candidate:
                            val_candidate = self.match_value(var_candidate, val_part)
                            if val_candidate:
                                conditions[var_candidate] = val_candidate
                                continue
                # Otherwise, try matching token as a value uniquely across variables.
                match_vars = []
                for v in self.cpnet.variables:
                    domain = self.domains.get(v, [])
                    for d in domain:
                        if d.lower() == token.lower():
                            match_vars.append(v)
                            break
                if len(match_vars) == 1:
                    v = match_vars[0]
                    val_candidate = self.match_value(v, token)
                    conditions[v] = val_candidate
                else:
                    raise ValueError(f"Ambiguous or no match for token '{token}'.")
        return conditions

    # ---------------- Domain and Variable Editing ----------------
    def edit_domain(self, variable):
        current_domain = self.domains.get(variable, [])
        current_str = ", ".join(current_domain)
        new_str = simpledialog.askstring("Edit Domain",
                                         f"Enter comma-separated values for {variable} (current: {current_str}):",
                                         initialvalue=current_str)
        if new_str is not None:
            new_domain = [x.strip() for x in new_str.split(",") if x.strip()]
            if new_domain:
                self.domains[variable] = new_domain
                self.update_cpt_display()
                self.update_graph()

    def edit_name(self, variable):
        new_name = simpledialog.askstring("Edit Name", f"Enter new name for variable {variable}:", initialvalue=variable)
        if new_name and new_name.strip():
            new_name = new_name.strip()
            idx = self.cpnet.variables.index(variable)
            self.cpnet.variables[idx] = new_name
            self.domains[new_name] = self.domains.pop(variable)
            self.cpnet.cpt[new_name] = self.cpnet.cpt.pop(variable)
            self.cpnet.cpt[new_name].variable = new_name
            for tbl in self.cpnet.cpt.values():
                for rule in tbl.rules:
                    if variable in rule.condition:
                        rule.condition[new_name] = rule.condition.pop(variable)
            self.update_all()

    # ---------------- Update Functions ----------------
    def update_all(self):
        self.update_cpt_display()
        self.update_graph()
        self.update_adv_dropdowns()
        self.update_seq_tree([])

    def update_adv_dropdowns(self):
        menu = self.head_dropdown_adv["menu"]
        menu.delete(0, tk.END)
        for var in self.cpnet.variables:
            menu.add_command(label=var, command=lambda v=var: self.head_var_adv.set(v))

    def update_cpt_display(self):
        for widget in self.cpt_display.winfo_children():
            widget.destroy()
        self.cpt_frames.clear()
        for var in self.cpnet.variables:
            domain = self.domains.get(var, [])
            header = f"{var} CP-table (Domain: {', '.join(domain)})"
            frame = tk.LabelFrame(self.cpt_display, text=header, padx=5, pady=5)
            frame.pack(fill="x", pady=3)
            btn_frame = tk.Frame(frame)
            btn_frame.grid(row=0, column=2, padx=5)
            tk.Button(btn_frame, text="Edit Domain", command=lambda v=var: self.edit_domain(v)).pack(side=tk.TOP, pady=2)
            tk.Button(btn_frame, text="Edit Name", command=lambda v=var: self.edit_name(v)).pack(side=tk.TOP, pady=2)
            tk.Label(frame, text="if", font=("Helvetica", 10, "bold"), borderwidth=1, relief="solid", width=20).grid(row=0, column=0, padx=1, pady=1)
            tk.Label(frame, text="then", font=("Helvetica", 10, "bold"), borderwidth=1, relief="solid", width=10).grid(row=0, column=1, padx=1, pady=1)
            if self.cpnet.cpt[var].rules:
                for idx, rule in enumerate(self.cpnet.cpt[var].rules, start=1):
                    cond_text = ", ".join(f"{k}={v}" for k, v in rule.condition.items()) if rule.condition else "Unconditional"
                    tk.Label(frame, text=cond_text, borderwidth=1, relief="solid", anchor="w", width=20).grid(row=idx, column=0, padx=1, pady=1)
                    order_text = f"{rule.ordering[0]}>{rule.ordering[1]}"
                    tk.Label(frame, text=order_text, borderwidth=1, relief="solid", anchor="w", width=10).grid(row=idx, column=1, padx=1, pady=1)
            else:
                tk.Label(frame, text="No rules", anchor="w").grid(row=1, column=0, columnspan=2, padx=1, pady=1)
            self.cpt_frames[var] = frame

    def update_graph(self):
        self.canvas.delete("all")
        n = len(self.cpnet.variables)
        if n == 0:
            return
        zoom = self.zoom_factor
        radius = 120 * zoom
        center_x, center_y = 150, 150  # Canvas is 300x300
        self.node_positions = {}
        for i, var in enumerate(self.cpnet.variables):
            angle = 2 * math.pi * i / n
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.node_positions[var] = (x, y)
            r = 12 * zoom
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="lightblue")
            self.canvas.create_text(x, y, text=var, font=("Helvetica", int(8*zoom), "bold"))
        for head in self.cpnet.variables:
            deps = self.cpnet.cpt[head].get_dependencies()
            for dep in deps:
                if dep in self.node_positions and head in self.node_positions:
                    x1, y1 = self.node_positions[dep]
                    x2, y2 = self.node_positions[head]
                    dx, dy = x2 - x1, y2 - y1
                    angle = math.atan2(dy, dx)
                    r_source = 12 * zoom
                    r_target = 12 * zoom
                    start_x = x1 + r_source * math.cos(angle)
                    start_y = y1 + r_source * math.sin(angle)
                    end_x = x2 - r_target * math.cos(angle)
                    end_y = y2 - r_target * math.sin(angle)
                    self.canvas.create_line(start_x, start_y, end_x, end_y, arrow=tk.LAST, width=2*zoom)

    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.update_graph()

    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.update_graph()

    def compute_sequence(self):
        try:
            source = self.parse_outcome_gui(self.src_entry.get())
            target = self.parse_outcome_gui(self.tgt_entry.get())
        except Exception as e:
            messagebox.showerror("Outcome Error", str(e))
            return
        sequence, exact = find_worsening_flipping_sequence(self.cpnet, source, target)
        self.update_seq_tree(sequence)

    def update_seq_tree(self, sequence):
        for item in self.seq_tree.get_children():
            self.seq_tree.delete(item)
        try:
            src = self.parse_outcome_gui(self.src_entry.get())
            src_str = ",".join(src[v] for v in self.cpnet.variables)
            self.seq_tree.insert("", "end", values=(src_str, "Source"), tags=("source",))
        except Exception:
            pass
        for step in sequence:
            var, justification, outcome_tuple = step
            outcome_str = ",".join(outcome_tuple)
            self.seq_tree.insert("", "end", values=(outcome_str, f"{var}: {justification}"), tags=(var,))
        self.seq_tree.tag_configure("source", background="#d0ffd0")

    def on_sequence_select(self, event):
        selected = self.seq_tree.focus()
        if not selected:
            return
        tags = self.seq_tree.item(selected, "tags")
        for frame in self.cpt_frames.values():
            frame.config(bg="SystemButtonFace")
        for tag in tags:
            if tag in self.cpt_frames:
                self.cpt_frames[tag].config(bg="yellow")
                break

    def add_advice(self):
        try:
            conditions = self.parse_advice_input(self.if_entry.get())
        except Exception as e:
            messagebox.showerror("Advice Input Error", str(e))
            return
        head = self.head_var_adv.get()
        ordering = ["1", "0"] if self.ordering_adv.get() == "1>0" else ["0", "1"]
        self.cpnet.cpt[head].add_rule(conditions, ordering)
        messagebox.showinfo("Advice Added", f"Rule added: if {conditions} then {ordering} for {head}")
        self.update_cpt_display()
        self.update_graph()
        self.if_entry.delete(0, tk.END)

    def update_all(self):
        self.update_cpt_display()
        self.update_graph()
        self.update_adv_dropdowns()
        self.update_seq_tree([])

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = CPNetGUI(root)
    app.run()
