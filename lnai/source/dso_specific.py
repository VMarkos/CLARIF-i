import os
import json
import time
import enum
import math
import hashlib
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# =====================================================================
# 1. Domain-Specific Language (DSL) Definition
# =====================================================================

class TokenType(enum.Enum):
    # Control Structure Tokens
    FOR_I = "FOR i IN 0..N-1"
    FOR_J_BUBBLE = "FOR j IN 0..N-i-2"      # Bound for Bubble Sort
    FOR_J_SELECTION = "FOR j IN i+1..N-1"   # Bound for Selection Sort
    IF_GT = "IF A[j] > A[j+1]"              # Bubble condition
    IF_GT_MIN = "IF A[j] < A[min_idx]"      # Selection condition
    
    # Action Primitives
    SWAP_ADJACENT = "SWAP(j, j+1)"
    SWAP_I_MIN = "SWAP(i, min_idx)"
    SET_MIN_J = "min_idx = j"
    
    # Block & Sequence Delimiters
    END_BLOCK = "END"
    STOP = "<STOP>"

DSL_TOKENS = list(TokenType)
TOKEN_TO_IDX = {token: idx for idx, token in enumerate(DSL_TOKENS)}
IDX_TO_TOKEN = {idx: token for idx, token in enumerate(DSL_TOKENS)}
VOCAB_SIZE = len(DSL_TOKENS)


# =====================================================================
# 2. Reference Algorithm Coaches
# =====================================================================

class AlgorithmCoach:
    @staticmethod
    def bubble_sort(arr):
        """Executes Bubble Sort and records the reference operational trace."""
        arr = list(arr)
        trace = []
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                trace.append(('COMPARE', j, j + 1))
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    trace.append(('SWAP', j, j + 1))
        return arr, trace

    @staticmethod
    def selection_sort(arr):
        """Executes Selection Sort and records the reference operational trace."""
        arr = list(arr)
        trace = []
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                trace.append(('COMPARE', j, min_idx))
                if arr[j] < arr[min_idx]:
                    min_idx = j
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                trace.append(('SWAP', i, min_idx))
        return arr, trace


# =====================================================================
# 3. Trace Distance Metric (Levenshtein)
# =====================================================================

def levenshtein_distance(seq1, seq2):
    """Computes Levenshtein edit distance between two primitive action sequences."""
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
# 4. AST / Virtual Machine Interpreter
# =====================================================================

class ProgramVirtualMachine:
    def __init__(self, max_op_steps=2500):
        self.max_op_steps = max_op_steps

    def execute(self, program_tokens, input_array):
        arr = list(input_array)
        n = len(arr)
        trace = []
        op_count = 0
        min_idx = 0
        
        # Build exact token execution flags
        has_bubble_loop = TokenType.FOR_J_BUBBLE in program_tokens
        has_select_loop = TokenType.FOR_J_SELECTION in program_tokens
        has_if_gt = TokenType.IF_GT in program_tokens
        has_if_min = TokenType.IF_GT_MIN in program_tokens
        has_swap_adj = TokenType.SWAP_ADJACENT in program_tokens
        has_swap_min = TokenType.SWAP_I_MIN in program_tokens
        has_set_min = TokenType.SET_MIN_J in program_tokens

        try:
            for i in range(n):
                if has_bubble_loop:
                    for j in range(0, max(0, n - i - 1)):
                        op_count += 1
                        if op_count > self.max_op_steps: break
                        
                        if has_if_gt:
                            trace.append(('COMPARE', j, j + 1))
                            if arr[j] > arr[j + 1]:
                                # ONLY SWAP IF SWAP TOKEN IS PRESENT
                                if has_swap_adj:
                                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                                    trace.append(('SWAP', j, j + 1))
                                elif has_set_min:
                                    min_idx = j

                elif has_select_loop:
                    min_idx = i
                    for j in range(i + 1, n):
                        op_count += 1
                        if op_count > self.max_op_steps: break
                        
                        if has_if_min:
                            trace.append(('COMPARE', j, min_idx))
                            if arr[j] < arr[min_idx]:
                                if has_set_min:
                                    min_idx = j
                    
                    if has_swap_min and min_idx != i:
                        arr[i], arr[min_idx] = arr[min_idx], arr[i]
                        trace.append(('SWAP', i, min_idx))

        except Exception:
            pass

        # Strict sort validation: must equal Python's sorted() AND must not be identical to original unsorted input (unless input was already sorted)
        is_sorted = (arr == sorted(input_array))
        return arr, trace, is_sorted

def get_valid_mask(current_tokens):
    """Returns a boolean mask of allowed next tokens based on grammar rules."""
    mask = torch.ones(VOCAB_SIZE, dtype=torch.bool)
    
    if not current_tokens:
        # Program MUST start with FOR_I
        mask[:] = False
        mask[TOKEN_TO_IDX[TokenType.FOR_I]] = True
        return mask
        
    last_token = current_tokens[-1]
    
    # Rule: FOR_I must be followed by an inner loop
    if last_token == TokenType.FOR_I:
        mask[:] = False
        mask[TOKEN_TO_IDX[TokenType.FOR_J_BUBBLE]] = True
        mask[TOKEN_TO_IDX[TokenType.FOR_J_SELECTION]] = True
        
    # Rule: Inner loop must be followed by IF condition
    elif last_token in (TokenType.FOR_J_BUBBLE, TokenType.FOR_J_SELECTION):
        mask[:] = False
        mask[TOKEN_TO_IDX[TokenType.IF_GT]] = True
        mask[TOKEN_TO_IDX[TokenType.IF_GT_MIN]] = True
        
    # Rule: IF condition MUST be followed by an Action (SWAP / SET)
    elif last_token in (TokenType.IF_GT, TokenType.IF_GT_MIN):
        mask[:] = False
        mask[TOKEN_TO_IDX[TokenType.SWAP_ADJACENT]] = True
        mask[TOKEN_TO_IDX[TokenType.SET_MIN_J]] = True
        
    # Rule: After SWAP/SET, program can STOP or add END blocks
    elif last_token in (TokenType.SWAP_ADJACENT, TokenType.SWAP_I_MIN, TokenType.SET_MIN_J):
        mask[:] = False
        mask[TOKEN_TO_IDX[TokenType.STOP]] = True
        
    return mask

# =====================================================================
# 5. DSO Controller Network
# =====================================================================

class DSOController(nn.Module):
    """RNN Controller that autoregressively generates symbolic AST tokens."""
    
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def sample_program(self, max_length=12):
        tokens = []
        log_probs = []
        
        hx = torch.zeros(1, self.hidden_dim)
        cx = torch.zeros(1, self.hidden_dim)
        
        # Fixed Start Token
        input_tok = torch.tensor([TOKEN_TO_IDX[TokenType.FOR_I]])
        tokens.append(IDX_TO_TOKEN[input_tok.item()])
        
        for _ in range(max_length - 1):
            embed = self.embedding(input_tok)
            hx, cx = self.lstm(embed, (hx, cx))
            logits = self.fc(hx)
            mask = get_valid_mask(tokens).unsqueeze(0)
            logits[~mask] = -1e9
            
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            log_probs.append(dist.log_prob(action))
            token = IDX_TO_TOKEN[action.item()]
            tokens.append(token)
            
            if token == TokenType.STOP:
                break
            input_tok = action

        return tokens, torch.stack(log_probs).sum()


# =====================================================================
# 6. Evaluation & Fitness
# =====================================================================

def validate_ast_structure(program_tokens):
    """Returns False if program has useless nesting or lacks required execution tokens."""
    has_loop = any(t in (TokenType.FOR_J_BUBBLE, TokenType.FOR_J_SELECTION) for t in program_tokens)
    has_condition = any(t in (TokenType.IF_GT, TokenType.IF_GT_MIN) for t in program_tokens)
    has_action = any(t in (TokenType.SWAP_ADJACENT, TokenType.SWAP_I_MIN, TokenType.SET_MIN_J) for t in program_tokens)
    
    # Check for excessive duplicate control tokens
    for_j_count = sum(1 for t in program_tokens if t in (TokenType.FOR_J_BUBBLE, TokenType.FOR_J_SELECTION))
    
    if not (has_loop and has_condition and has_action) or for_j_count > 2:
        return False
    return True

def evaluate_program(program_tokens, coach_type="bubble", test_cases=35, array_len=5):
    """Evaluates program sorting accuracy and coach trace similarity."""
    if not validate_ast_structure(program_tokens):
        return -1.0, 0.0, 0.0
    vm = ProgramVirtualMachine()
    total_sort_success = 0
    fidelities = []
    
    for _ in range(test_cases):
        arr = np.random.permutation(array_len).tolist()
        
        if coach_type == "bubble":
            _, coach_trace = AlgorithmCoach.bubble_sort(arr)
        elif coach_type == "selection":
            _, coach_trace = AlgorithmCoach.selection_sort(arr)
        else:
            raise ValueError(f"Unknown coach type: {coach_type}")
            
        _, prog_trace, is_sorted = vm.execute(program_tokens, arr)
        
        has_swaps = any(event[0] == 'SWAP' for event in prog_trace)

        if is_sorted and (has_swaps or arr == sorted(arr)):
            total_sort_success += 1
            
        dist = levenshtein_distance(prog_trace, coach_trace)
        max_len = max(len(prog_trace), len(coach_trace), 1)
        fidelity = 1.0 - (dist / max_len)
        fidelities.append(fidelity)
        
    accuracy = total_sort_success / test_cases
    mean_fidelity = float(np.mean(fidelities))
    reward = 0.5 * accuracy + 0.5 * mean_fidelity
    return reward, accuracy, mean_fidelity


# =====================================================================
# 7. Unique Identifier & Experiment Setup
# =====================================================================

def generate_run_id(params: dict) -> str:
    """Generates a unique identifier string encoding parametrisation + timestamp."""
    param_str = "_".join([f"{k}-{v}" for k, v in sorted(params.items())])
    hash_digest = hashlib.md5(param_str.encode("utf-8")).hexdigest()[:8]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"dso_{param_str}_{timestamp}_{hash_digest}"


# =====================================================================
# 8. Training & Testing Routine
# =====================================================================

def train_and_evaluate_dso(
    epochs=120,
    batch_size=16,
    lr=0.003,
    quantile=0.7,
    coach_type="bubble",
    train_array_len=5,
    save_dir="./dso_results"
):
    params = {
        "coach": coach_type,
        "epochs": epochs,
        "bs": batch_size,
        "lr": lr,
        "q": quantile,
        "n": train_array_len
    }
    
    run_id = generate_run_id(params)
    run_dir = os.path.join(save_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"==================================================")
    print(f" Starting Experiment: {run_id}")
    print(f" Saving Artifacts To: {run_dir}")
    print(f"==================================================\n")
    
    controller = DSOController(VOCAB_SIZE)
    optimizer = optim.Adam(controller.parameters(), lr=lr)
    
    history = {
        "epoch": [],
        "mean_reward": [],
        "max_reward": [],
        "mean_accuracy": [],
        "mean_fidelity": []
    }
    
    best_program = None
    best_reward = -float("inf")
    
    for epoch in range(1, epochs + 1):
        batch_log_probs = []
        batch_rewards = []
        batch_accs = []
        batch_fids = []
        batch_programs = []
        
        for _ in range(batch_size):
            program, log_prob = controller.sample_program()
            reward, acc, fid = evaluate_program(
                program, coach_type=coach_type, test_cases=35, array_len=train_array_len
            )
            
            batch_log_probs.append(log_prob)
            batch_rewards.append(reward)
            batch_accs.append(acc)
            batch_fids.append(fid)
            batch_programs.append(program)
            
            if reward > best_reward:
                best_reward = reward
                best_program = program
                
        # Risk-Seeking Baseline calculation
        baseline = np.quantile(batch_rewards, quantile)
        
        policy_loss = 0
        for log_prob, r in zip(batch_log_probs, batch_rewards):
            advantage = r - baseline
            policy_loss += -log_prob * advantage
            
        policy_loss = policy_loss / batch_size
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # Logging history
        history["epoch"].append(epoch)
        history["mean_reward"].append(float(np.mean(batch_rewards)))
        history["max_reward"].append(float(np.max(batch_rewards)))
        history["mean_accuracy"].append(float(np.mean(batch_accs)))
        history["mean_fidelity"].append(float(np.mean(batch_fids)))
        
        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs} | "
                  f"Max Reward: {np.max(batch_rewards):.3f} | "
                  f"Avg Acc: {np.mean(batch_accs)*100:.1f}% | "
                  f"Avg Fidelity: {np.mean(batch_fids)*100:.1f}%")
            readable_ast = [t.value for t in best_program if t != TokenType.STOP]
            print(f"  Best Program AST: {readable_ast}\n")

    # Save Model Weights
    model_path = os.path.join(run_dir, "controller_model.pt")
    torch.save(controller.state_dict(), model_path)
    
    # Run Generalizability Test across out-of-sample array lengths
    test_lengths = [3, 4, 5, 6, 7, 8]
    test_results = {}
    print("--- Running Out-of-Sample Generalizability Tests ---")
    for test_n in test_lengths:
        reward, acc, fid = evaluate_program(
            best_program, coach_type=coach_type, test_cases=35, array_len=test_n
        )
        test_results[f"N={test_n}"] = {"accuracy": acc, "fidelity": fid, "reward": reward}
        print(f" Array Length N={test_n} -> Sorting Acc: {acc*100:.1f}%, Trace Fidelity: {fid*100:.1f}%")

    # Save Metrics JSON
    best_ast_str = [t.value for t in best_program if t != TokenType.STOP]
    results_payload = {
        "run_id": run_id,
        "parametrisation": params,
        "best_program_ast": best_ast_str,
        "best_training_reward": float(best_reward),
        "test_generalizability": test_results,
        "training_history": history
    }
    
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_payload, f, indent=4)
        
    # Plotting Training Dynamics and Test Generalizability
    plot_path = os.path.join(run_dir, "training_and_test_plots.png")
    plot_experiment_results(history, test_results, run_id, plot_path)
    
    print(f"\nExperiment complete. All assets saved to {run_dir}")
    return results_payload


# =====================================================================
# 9. Plotting Utilities
# =====================================================================

def plot_experiment_results(history, test_results, run_id, save_path):
    """Generates dual-panel convergence and generalizability plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Training Convergence Curves
    epochs = history["epoch"]
    ax1.plot(epochs, history["mean_reward"], label="Mean Reward", color="tab:blue")
    ax1.plot(epochs, history["max_reward"], label="Max Reward", color="tab:green", linestyle="--")
    ax1.plot(epochs, history["mean_accuracy"], label="Sorting Accuracy", color="tab:orange", alpha=0.7)
    ax1.plot(epochs, history["mean_fidelity"], label="Trace Fidelity", color="tab:purple", alpha=0.7)
    
    ax1.set_title("DSO Training Convergence")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Metric Score [0, 1]")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")
    
    # Panel 2: Test Generalizability across Array Sizes
    lengths = list(test_results.keys())
    accs = [test_results[k]["accuracy"] for k in lengths]
    fids = [test_results[k]["fidelity"] for k in lengths]
    
    x = np.arange(len(lengths))
    width = 0.35
    
    ax2.bar(x - width/2, accs, width, label="Sorting Accuracy", color="tab:orange")
    ax2.bar(x + width/2, fids, width, label="Trace Fidelity", color="tab:purple")
    
    ax2.set_title("Program Generalizability Out-of-Sample")
    ax2.set_xlabel("Test Array Length")
    ax2.set_ylabel("Score [0, 1]")
    ax2.set_xticks(x)
    ax2.set_xticklabels(lengths)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left")
    
    plt.suptitle(f"Run ID: {run_id}", fontsize=10, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Example Run: Train DSO on Bubble Sort target
    results = train_and_evaluate_dso(
        epochs=100,
        batch_size=16,
        lr=0.0025,
        quantile=0.85,
        coach_type="bubble",
        train_array_len=6,
        save_dir="./dso_results"
    )
