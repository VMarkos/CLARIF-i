import enum
import json
import os
import random
import uuid
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# =====================================================================
# 1. DSL Tokens & Grammar Masking
# =====================================================================

class TokenType(enum.Enum):
    LET_CTR = "LET_CTR"       # LET_CTR <var> <val>
    SET_CTR = "SET_CTR"       # SET_CTR <var1> <var2>
    STEP_CTR = "STEP_CTR"     # STEP_CTR <var> <dir>
    
    WHILE = "WHILE"           # WHILE <var1> <op> <var2>
    WHILE_END = "WHILE_END"
    IF = "IF"                 # IF A[<var1>] <op> A[<var2>]
    IF_END = "IF_END"
    
    OP_LT = "<"
    OP_GT = ">"
    OP_EQ = "=="
    OP_INC = "+1"
    OP_DEC = "-1"
    
    REG_I = "i"
    REG_J = "j"
    REG_MIN = "min_ptr"
    REG_N = "N"
    REG_0 = "0"
    REG_1 = "1"
    
    SWAP = "SWAP"             # SWAP A[<var1>], A[<var2>]
    STOP = "<STOP>"

DSL_TOKENS = list(TokenType)
TOKEN_TO_IDX = {token: idx for idx, token in enumerate(DSL_TOKENS)}
IDX_TO_TOKEN = {idx: token for idx, token in enumerate(DSL_TOKENS)}
VOCAB_SIZE = len(DSL_TOKENS)

WRITABLE_REGS = {TokenType.REG_I, TokenType.REG_J, TokenType.REG_MIN}
ALL_REGS = {TokenType.REG_I, TokenType.REG_J, TokenType.REG_MIN, TokenType.REG_N, TokenType.REG_0, TokenType.REG_1}

def get_initialized_registers(current_tokens):
    initialized = {"N", "0", "1"}
    for k in range(len(current_tokens) - 2):
        if current_tokens[k] == TokenType.LET_CTR:
            var_token = current_tokens[k + 1]
            if var_token in WRITABLE_REGS:
                initialized.add(var_token.value)
    return initialized

def get_valid_mask(current_tokens, min_length=8):
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    
    while_depth = current_tokens.count(TokenType.WHILE) - current_tokens.count(TokenType.WHILE_END)
    if_depth = current_tokens.count(TokenType.IF) - current_tokens.count(TokenType.IF_END)
    init_vars = get_initialized_registers(current_tokens)

    if not current_tokens:
        mask[TOKEN_TO_IDX[TokenType.LET_CTR]] = True
        return mask

    last = current_tokens[-1]
    penultimate = current_tokens[-2] if len(current_tokens) >= 2 else None
    antepenultimate = current_tokens[-3] if len(current_tokens) >= 3 else None

    if len(current_tokens) >= 22:
        if if_depth > 0:
            mask[TOKEN_TO_IDX[TokenType.IF_END]] = True
        elif while_depth > 0:
            mask[TOKEN_TO_IDX[TokenType.WHILE_END]] = True
        else:
            mask[TOKEN_TO_IDX[TokenType.STOP]] = True
        return mask

    def enable_initialized_readable():
        for var in init_vars:
            mask[TOKEN_TO_IDX[TokenType(var)]] = True

    def enable_writable_regs():
        for r in WRITABLE_REGS:
            mask[TOKEN_TO_IDX[r]] = True

    if last == TokenType.LET_CTR:
        enable_writable_regs()
        return mask
    elif penultimate == TokenType.LET_CTR and last in WRITABLE_REGS:
        enable_initialized_readable()
        return mask

    elif last == TokenType.SET_CTR:
        enable_writable_regs()
        return mask
    elif penultimate == TokenType.SET_CTR and last in WRITABLE_REGS:
        enable_initialized_readable()
        return mask

    elif last == TokenType.STEP_CTR:
        enable_writable_regs()
        return mask
    elif penultimate == TokenType.STEP_CTR and last in WRITABLE_REGS:
        mask[TOKEN_TO_IDX[TokenType.OP_INC]] = True
        mask[TOKEN_TO_IDX[TokenType.OP_DEC]] = True
        return mask

    elif last == TokenType.WHILE:
        enable_initialized_readable()
        return mask
    elif penultimate == TokenType.WHILE and last in ALL_REGS:
        mask[TOKEN_TO_IDX[TokenType.OP_LT]] = True
        mask[TOKEN_TO_IDX[TokenType.OP_GT]] = True
        mask[TOKEN_TO_IDX[TokenType.OP_EQ]] = True
        return mask
    elif antepenultimate == TokenType.WHILE and penultimate in (TokenType.OP_LT, TokenType.OP_GT, TokenType.OP_EQ):
        enable_initialized_readable()
        return mask

    elif last == TokenType.IF:
        enable_initialized_readable()
        return mask
    elif penultimate == TokenType.IF and last in ALL_REGS:
        mask[TOKEN_TO_IDX[TokenType.OP_LT]] = True
        mask[TOKEN_TO_IDX[TokenType.OP_GT]] = True
        mask[TOKEN_TO_IDX[TokenType.OP_EQ]] = True
        return mask
    elif antepenultimate == TokenType.IF and penultimate in (TokenType.OP_LT, TokenType.OP_GT, TokenType.OP_EQ):
        enable_initialized_readable()
        return mask

    elif last == TokenType.SWAP:
        enable_initialized_readable()
        return mask
    elif penultimate == TokenType.SWAP and last in ALL_REGS:
        enable_initialized_readable()
        return mask

    mask[TOKEN_TO_IDX[TokenType.LET_CTR]] = True
    mask[TOKEN_TO_IDX[TokenType.SET_CTR]] = True
    mask[TOKEN_TO_IDX[TokenType.STEP_CTR]] = True
    mask[TOKEN_TO_IDX[TokenType.WHILE]] = True
    mask[TOKEN_TO_IDX[TokenType.IF]] = True
    mask[TOKEN_TO_IDX[TokenType.SWAP]] = True
    
    if while_depth > 0:
        mask[TOKEN_TO_IDX[TokenType.WHILE_END]] = True
    if if_depth > 0:
        mask[TOKEN_TO_IDX[TokenType.IF_END]] = True
    if while_depth == 0 and if_depth == 0:
        mask[TOKEN_TO_IDX[TokenType.STOP]] = True

    # Only allow STOP if we reached minimum program length AND all loops/conditionals are closed
    if len(current_tokens) >= min_length and while_depth == 0 and if_depth == 0:
        mask[TOKEN_TO_IDX[TokenType.STOP]] = True
    else:
        mask[TOKEN_TO_IDX[TokenType.STOP]] = False

    return mask

# =====================================================================
# 2. Virtual Machine & Evaluation Metrics
# =====================================================================

class VMExecutionError(Exception):
    pass

class StrictImperativeVM:
    def __init__(self, max_op_steps=2000):
        self.max_op_steps = max_op_steps

    def parse_instructions(self, tokens):
        instructions = []
        idx = 0
        while idx < len(tokens):
            tok = tokens[idx]
            if tok == TokenType.LET_CTR and idx + 2 < len(tokens):
                instructions.append(('LET', tokens[idx+1].value, tokens[idx+2].value))
                idx += 3
            elif tok == TokenType.SET_CTR and idx + 2 < len(tokens):
                instructions.append(('SET', tokens[idx+1].value, tokens[idx+2].value))
                idx += 3
            elif tok == TokenType.STEP_CTR and idx + 2 < len(tokens):
                instructions.append(('STEP', tokens[idx+1].value, tokens[idx+2].value))
                idx += 3
            elif tok == TokenType.WHILE and idx + 3 < len(tokens):
                instructions.append(('WHILE', tokens[idx+1].value, tokens[idx+2].value, tokens[idx+3].value))
                idx += 4
            elif tok == TokenType.WHILE_END:
                instructions.append(('WHILE_END',))
                idx += 1
            elif tok == TokenType.IF and idx + 3 < len(tokens):
                instructions.append(('IF', tokens[idx+1].value, tokens[idx+2].value, tokens[idx+3].value))
                idx += 4
            elif tok == TokenType.IF_END:
                instructions.append(('IF_END',))
                idx += 1
            elif tok == TokenType.SWAP and idx + 2 < len(tokens):
                instructions.append(('SWAP', tokens[idx+1].value, tokens[idx+2].value))
                idx += 3
            else:
                idx += 1
        return instructions

    def execute(self, tokens, input_array):
        arr = list(input_array)
        n = len(arr)
        registers = {'i': None, 'j': None, 'min_ptr': None, 'N': n, '0': 0, '1': 1}
        trace, op_count, pc = [], 0, 0
        
        instructions = self.parse_instructions(tokens)

        loop_stack, loop_map = [], {}
        for i, inst in enumerate(instructions):
            if inst[0] == 'WHILE':
                loop_stack.append(i)
            elif inst[0] == 'WHILE_END' and loop_stack:
                start = loop_stack.pop()
                loop_map[start] = i
                loop_map[i] = start

        def get_reg(v):
            val = registers.get(v, None)
            if val is None:
                raise VMExecutionError(f"Uninitialized register '{v}'")
            return val

        try:
            while pc < len(instructions):
                op_count += 1
                if op_count > self.max_op_steps:
                    raise VMExecutionError("Infinite Loop")

                inst = instructions[pc]
                op = inst[0]
                
                if op in ('LET', 'SET'):
                    registers[inst[1]] = get_reg(inst[2])
                elif op == 'STEP':
                    registers[inst[1]] = get_reg(inst[1]) + (1 if inst[2] == '+1' else -1)
                elif op == 'WHILE':
                    v1, v2 = get_reg(inst[1]), get_reg(inst[3])
                    cond = (v1 < v2) if inst[2] == '<' else (v1 > v2 if inst[2] == '>' else v1 == v2)
                    if not cond:
                        pc = loop_map.get(pc, len(instructions))
                elif op == 'WHILE_END':
                    pc = loop_map.get(pc, pc) - 1
                elif op == 'IF':
                    idx1, idx2 = get_reg(inst[1]), get_reg(inst[3])
                    if not (0 <= idx1 < n and 0 <= idx2 < n):
                        raise VMExecutionError(f"IF bounds error: ({idx1}, {idx2})")
                    trace.append(('COMPARE', idx1, idx2))
                    v1, v2 = arr[idx1], arr[idx2]
                    cond = (v1 < v2) if inst[2] == '<' else (v1 > v2 if inst[2] == '>' else v1 == v2)
                    if not cond:
                        depth = 1
                        while pc + 1 < len(instructions) and depth > 0:
                            pc += 1
                            if instructions[pc][0] == 'IF': depth += 1
                            elif instructions[pc][0] == 'IF_END': depth -= 1
                elif op == 'SWAP':
                    idx1, idx2 = get_reg(inst[1]), get_reg(inst[2])
                    if not (0 <= idx1 < n and 0 <= idx2 < n):
                        raise VMExecutionError(f"SWAP bounds error: ({idx1}, {idx2})")
                    if idx1 != idx2:
                        arr[idx1], arr[idx2] = arr[idx2], arr[idx1]
                        trace.append(('SWAP', idx1, idx2))
                pc += 1

        except VMExecutionError:
            return arr, trace, False, False

        is_sorted = (arr == sorted(input_array))
        return arr, trace, True, is_sorted

class ImperativeCoach:
    @staticmethod
    def bubble_sort(arr):
        arr, trace, n = list(arr), [], len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                trace.append(('COMPARE', j, j + 1))
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    trace.append(('SWAP', j, j + 1))
        return arr, trace

def levenshtein_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

# =====================================================================
# 3. Controller Network & Evaluator
# =====================================================================

class GenericImperativeController(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def sample_program(self, max_length=24, greedy=False):
        tokens, log_probs = [], []
        hx = torch.zeros(1, self.hidden_dim)
        cx = torch.zeros(1, self.hidden_dim)
        
        input_tok = torch.tensor([TOKEN_TO_IDX[TokenType.LET_CTR]])
        tokens.append(IDX_TO_TOKEN[input_tok.item()])
        
        for _ in range(max_length - 1):
            embed = self.embedding(input_tok)
            hx, cx = self.lstm(embed, (hx, cx))
            logits = self.fc(hx).squeeze(0)
            
            mask = get_valid_mask(tokens)
            logits[~mask] = -1e9
            
            if greedy:
                action = torch.argmax(logits)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                
            token = IDX_TO_TOKEN[action.item()]
            tokens.append(token)
            
            if token == TokenType.STOP:
                break
            input_tok = action.unsqueeze(0)

        sum_log_prob = torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0)
        return tokens, sum_log_prob

    def sample_program_with_entropy(self, max_length=24, min_length=8):
        tokens = []
        log_probs = []
        entropies = []
        
        hx = torch.zeros(1, self.hidden_dim)
        cx = torch.zeros(1, self.hidden_dim)
        
        input_tok = torch.tensor([TOKEN_TO_IDX[TokenType.LET_CTR]])
        tokens.append(IDX_TO_TOKEN[input_tok.item()])
        
        for _ in range(max_length - 1):
            embed = self.embedding(input_tok)
            hx, cx = self.lstm(embed, (hx, cx))
            logits = self.fc(hx).squeeze(0)
            
            # Enforce grammar mask (including minimum length constraint)
            mask = get_valid_mask(tokens, min_length=min_length)
            logits[~mask] = -1e9
            
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            
            token = IDX_TO_TOKEN[action.item()]
            tokens.append(token)
            
            if token == TokenType.STOP:
                break
            input_tok = action.unsqueeze(0)

        sum_log_prob = torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0)
        sum_entropy = torch.stack(entropies).sum() if entropies else torch.tensor(0.0)
        
        return tokens, sum_log_prob, sum_entropy

def evaluate_program(program_tokens, coach_type="bubble", test_cases=10, array_len=6):
    vm = StrictImperativeVM()
    total_sort_success, clean_exec_count, fidelities = 0, 0, []
    
    for _ in range(test_cases):
        arr = np.random.permutation(array_len).tolist()
        _, coach_trace = ImperativeCoach.bubble_sort(arr)
        _, prog_trace, clean_exec, is_sorted = vm.execute(program_tokens, arr)
        
        if clean_exec: clean_exec_count += 1
        if is_sorted: total_sort_success += 1
            
        dist = levenshtein_distance(prog_trace, coach_trace)
        max_len = max(len(prog_trace), len(coach_trace), 1)
        fidelities.append(1.0 - (dist / max_len))

    accuracy = total_sort_success / test_cases
    exec_ratio = clean_exec_count / test_cases
    mean_fidelity = float(np.mean(fidelities))
    reward = (0.2 * exec_ratio) + (0.3 * mean_fidelity) + (0.5 * accuracy)
    return reward, accuracy, mean_fidelity

# =====================================================================
# 4. Printing & Testing Utilities
# =====================================================================

def pretty_print_ast(program_tokens, title="Synthesized Imperative Algorithm"):
    """Formats raw token sequence into structured, indented Python pseudo-code."""
    vm = StrictImperativeVM()
    instructions = vm.parse_instructions(program_tokens)
    
    print("\n" + "=" * 55)
    print(f" {title.upper()}")
    print("=" * 55)
    print("def synthesized_sort(A, N):")
    
    indent = 1
    indent_str = "    "
    
    for inst in instructions:
        op = inst[0]
        if op == "LET":
            print(f"{indent_str * indent}{inst[1]} = {inst[2]}")
        elif op == "SET":
            print(f"{indent_str * indent}{inst[1]} = {inst[2]}")
        elif op == "STEP":
            op_symbol = "+=" if inst[2] == "+1" else "-="
            print(f"{indent_str * indent}{inst[1]} {op_symbol} 1")
        elif op == "WHILE":
            print(f"{indent_str * indent}while {inst[1]} {inst[2]} {inst[3]}:")
            indent += 1
        elif op == "WHILE_END":
            indent = max(1, indent - 1)
        elif op == "IF":
            print(f"{indent_str * indent}if A[{inst[1]}] {inst[2]} A[{inst[3]}]:")
            indent += 1
        elif op == "IF_END":
            indent = max(1, indent - 1)
        elif op == "SWAP":
            print(f"{indent_str * indent}A[{inst[1]}], A[{inst[2]}] = A[{inst[2]}], A[{inst[1]}]")
            
    print("=" * 55 + "\n")

def test_saved_model(save_dir, test_n_values=[3, 5, 10, 15, 20], num_trials=50):
    """Loads a saved controller and tests OOD generalization across array sizes."""
    model_path = os.path.join(save_dir, "model.pt")
    controller = GenericImperativeController(VOCAB_SIZE)
    controller.load_state_dict(torch.load(model_path, map_location="cpu"))
    controller.eval()

    best_program, _ = controller.sample_program(max_length=24, greedy=True)
    
    pretty_print_ast(best_program, title="Loaded Model Extracted AST")

    vm = StrictImperativeVM()
    results = {}
    
    print(f"=== OOD Benchmarking Across Array Lengths (N) ===")
    for n in test_n_values:
        sort_successes = 0
        clean_execs = 0
        
        for _ in range(num_trials):
            arr = np.random.permutation(n).tolist()
            _, _, clean_exec, is_sorted = vm.execute(best_program, arr)
            if clean_exec: clean_execs += 1
            if is_sorted: sort_successes += 1
                
        acc = (sort_successes / num_trials) * 100
        exec_acc = (clean_execs / num_trials) * 100
        results[n] = acc
        print(f"  N = {n:02d} | Sorting Acc: {acc:5.1f}% | Execution Safety: {exec_acc:5.1f}%")

    plot_generalization_curve(results, save_dir)
    return results

def plot_generalization_curve(results, save_dir):
    """Plots Out-Of-Distribution (OOD) generalization performance."""
    n_vals = list(results.keys())
    acc_vals = list(results.values())
    
    plt.figure(figsize=(7, 4))
    plt.plot(n_vals, acc_vals, marker='o', color='tab:green', linewidth=2, label="Sort Accuracy (%)")
    plt.title("OOD Generalization vs Array Size (N)", fontsize=11, fontweight="bold")
    plt.xlabel("Array Length (N)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(-5, 105)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    plot_path = os.path.join(save_dir, "ood_generalization.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] OOD Plot -> {plot_path}")

def plot_training_curve(history, save_dir, run_name):
    """Plots training max and mean reward convergence."""
    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["max_reward"], label="Max Reward", color="tab:blue")
    plt.plot(history["epoch"], history["mean_reward"], label="Mean Reward", color="tab:orange", linestyle="--")
    plt.title(f"Training Convergence: {run_name}", fontsize=10, fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    plot_path = os.path.join(save_dir, "training_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Training Curve -> {plot_path}")

# =====================================================================
# 5. Training Pipeline
# =====================================================================

def train_and_save_imperative_dso(coach_type="bubble", epochs=100, batch_size=64, lr=0.003):
    unique_hash = uuid.uuid4().hex[:6]
    run_name = f"imperative_{coach_type}_lr{lr}_bs{batch_size}_ep{epochs}_{unique_hash}"
    save_dir = os.path.join("./dso_imperative", run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    controller = GenericImperativeController(VOCAB_SIZE)
    optimizer = optim.Adam(controller.parameters(), lr=lr)
    
    history = {"epoch": [], "max_reward": [], "mean_reward": []}
    
    print(f"=== Training Strict Imperative DSO [{run_name}] ===")
    
    for epoch in range(1, epochs + 1):
        batch_log_probs, batch_entropies, batch_rewards = [], [], []
        
        for _ in range(batch_size):
            program, log_prob, entropy = controller.sample_program_with_entropy()
            r, _, _ = evaluate_program(program, coach_type=coach_type)
            batch_log_probs.append(log_prob)
            batch_entropies.append(entropy)
            batch_rewards.append(r)

        baseline = np.mean(batch_rewards)
        entropy_coeff = 0.01
        policy_loss = sum([-lp * (r - baseline) - entropy_coeff * ent for lp, ent, r in zip(batch_log_probs, batch_entropies, batch_rewards)]) / batch_size
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        history["epoch"].append(epoch)
        history["max_reward"].append(float(np.max(batch_rewards)))
        history["mean_reward"].append(float(np.mean(batch_rewards)))
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f" Epoch {epoch:03d}/{epochs} | Max Reward: {np.max(batch_rewards):.3f} | Mean Reward: {np.mean(batch_rewards):.3f}")

    model_path = os.path.join(save_dir, "model.pt")
    history_path = os.path.join(save_dir, "history.json")
    
    torch.save(controller.state_dict(), model_path)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"\n[Saved] Checkpoint -> {model_path}")
    print(f"[Saved] Metrics -> {history_path}")

    plot_training_curve(history, save_dir, run_name)
    return save_dir

# =====================================================================
# 6. Main Execution Pipeline
# =====================================================================

if __name__ == "__main__":
    # 1. Train DSO Controller
    save_dir = train_and_save_imperative_dso(
        coach_type="bubble",
        epochs=500,
        batch_size=64,
        lr=0.001
    )

    # 2. Run Benchmarking & AST Printing Utilities
    test_saved_model(
        save_dir=save_dir,
        test_n_values=[3, 5, 8, 10, 15, 20],
        num_trials=50
    )
