# engines.py
import copy

def parse_assignment(text):
    assignments = {}
    text = text.strip()
    if not text or text.lower().startswith("e.g."):
        return assignments
    for part in text.split(";"):
        part = part.strip()
        if part and "=" in part:
            var, val = part.split("=", 1)
            assignments[var.strip()] = val.strip()
    return assignments

def parse_variable_domains(text):
    domains = {}
    for part in text.split(";"):
        part = part.strip()
        if part and ":" in part:
            var, rest = part.split(":", 1)
            var = var.strip()
            rest = rest.strip().strip("{}")
            values = [val.strip() for val in rest.split(",") if val.strip()]
            domains[var] = set(values)
    return domains

def is_target_reached(current_state, target_state):
    for k, v in target_state.items():
        if current_state.get(k) != v:
            return False
    return True

def rule_applicable(rule, state):
    for k, v in rule["condition"].items():
        if state.get(k) != v:
            return False
    return True

def apply_rule(rule, state):
    new_state = copy.deepcopy(state)
    for k, v in rule["preference"].items():
        new_state[k] = v
    return new_state

def dfs_collect_traces(current_state, target_state, rules, visited, trace, collected):
    score = sum(1 for k, v in target_state.items() if current_state.get(k) == v)
    if is_target_reached(current_state, target_state):
        new_trace = trace + [(current_state, "Target Reached")]
        collected.append((new_trace, score))
        return

    applicable = [r for r in rules if rule_applicable(r, current_state)]
    accepted_found = False
    for r in applicable:
        head_var = next(iter(r["preference"]))
        rule_value = r["preference"][head_var]
        defeated = False
        for r2 in rules:
            if r2["rule_name"] == r["rule_name"]:
                continue
            if next(iter(r2["preference"])) != head_var:
                continue
            if not rule_applicable(r2, current_state):
                continue
            if r2.get("priority", 0) > r.get("priority", 0) and r2["preference"][head_var] != rule_value:
                defeated = True
                break
        if not defeated:
            accepted_found = True
            break

    new_trace = trace + [(current_state, "No move")]
    if not accepted_found:
        collected.append((new_trace, score))
    
    state_key = frozenset(current_state.items())
    if state_key in visited:
        return
    visited.add(state_key)
    for r in applicable:
        new_state = apply_rule(r, current_state)
        dfs_collect_traces(new_state, target_state, rules, visited, trace + [(current_state, r["rule_name"])], collected)
    visited.remove(state_key)

def compute_statuses(rules, trace):
    status_table = {}
    for r in rules:
        rule_name = r["rule_name"]
        status_table[rule_name] = {}
        for i, (state, transition) in enumerate(trace):
            if not rule_applicable(r, state):
                status_table[rule_name][i] = "rejected"
            elif transition == r["rule_name"]:
                status_table[rule_name][i] = "selected"
            else:
                head_var = next(iter(r["preference"]))
                rule_value = r["preference"][head_var]
                defeated = False
                for r2 in rules:
                    if r2["rule_name"] == r["rule_name"]:
                        continue
                    if next(iter(r2["preference"])) != head_var:
                        continue
                    if not rule_applicable(r2, state):
                        continue
                    if r2.get("priority", 0) > r.get("priority", 0) and r2["preference"][head_var] != rule_value:
                        defeated = True
                        break
                if defeated:
                    status_table[rule_name][i] = "defeated"
                else:
                    status_table[rule_name][i] = "accepted"
    return status_table
