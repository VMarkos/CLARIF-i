import enum
import hashlib
import json
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# =====================================================================
# 1. DSL & Reference Coach Definitions
# =====================================================================

class TokenType(enum.Enum):
    FOR_I = "FOR i IN 0..N-1"
    FOR_J_BUBBLE = "FOR j IN 0..N-i-2"
    FOR_J_SELECTION = "FOR j IN i+1..N-1"
    IF_GT = "IF A[j] > A[j+1]"
    IF_GT_MIN = "IF A[j] < A[min_idx]"
    SWAP_ADJACENT = "SWAP(j, j+1)"
    SWAP_I_MIN = "SWAP(i, min_idx)"
    SET_MIN_J = "min_idx = j"
    END_BLOCK = "END"
    STOP = "<STOP>"

DSL_TOKENS = list(TokenType)
TOKEN_TO_IDX = {token: idx for idx, token in enumerate(DSL_TOKENS)}
IDX_TO_TOKEN = {idx: token for idx, token in enumerate(DSL_TOKENS)}
VOCAB_SIZE = len(DSL_TOKENS)


class AlgorithmCoach:
    @staticmethod
    def bubble_sort(arr):
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
# 2. Virtual Machine (Scaled step limit for N=20)
# =====================================================================

class ProgramVirtualMachine:
    def __init__(self, max_op_steps=2500):  # Expanded bound for O(N^2) at N=20
        self.max_op_steps = max_op_steps

    def execute(self, program_tokens, input_array):
        arr = list(input_array)
        n = len(arr)
        trace = []
        op_count = 0
        
        idx = 0
        try:
            while idx < len(program_tokens):
                token = program_tokens[idx]
                if token == TokenType.FOR_I:
                    for i in range(n):
                        if TokenType.FOR_J_BUBBLE in program_tokens:
                            for j_val in range(0, max(0, n - i - 1)):
                                op_count += 1
                                if op_count > self.max_op_steps: break
                                
                                if TokenType.IF_GT in program_tokens:
                                    trace.append(('COMPARE', j_val, j_val + 1))
                                    if arr[j_val] > arr[j_val + 1]:
                                        arr[j_val], arr[j_val + 1] = arr[j_val + 1], arr[j_val]
                                        trace.append(('SWAP', j_val, j_val + 1))
                        if op_count > self.max_op_steps: break
                    break
                idx += 1
        except Exception:
            pass

        is_sorted = (arr == sorted(input_array))
        return arr, trace, is_sorted


# =====================================================================
# 3. DSO Controller Network
# =====================================================================

class DSOController(nn.Module):
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
        
        input_tok = torch.tensor([TOKEN_TO_IDX[TokenType.FOR_I]])
        tokens.append(IDX_TO_TOKEN[input_tok.item()])
        
        for _ in range(max_length - 1):
            embed = self.embedding(input_tok)
            hx, cx = self.lstm(embed, (hx, cx))
            logits = self.fc(hx)
            
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
# 4. Evaluation Function
# =====================================================================

def evaluate_program(program_tokens, coach_type="bubble", test_cases=25, array_len=20, vm_max_steps=2500):
    vm = ProgramVirtualMachine(max_op_steps=vm_max_steps)
    total_sort_success = 0
    fidelities = []
    
    for _ in range(test_cases):
        arr = np.random.permutation(array_len).tolist()
        _, coach_trace = AlgorithmCoach.bubble_sort(arr)
        _, prog_trace, is_sorted = vm.execute(program_tokens, arr)
        
        if is_sorted:
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
# 5. Curriculum Pre-Training & Transfer Pipeline
# =====================================================================

def pretrain_controller(
    controller,
    optimizer,
    pretrain_epochs=50,
    batch_size=32,
    quantile=0.75,
    pretrain_n=5
):
    """Phase 1: Pre-trains controller on smaller N=5 to discover basic DSL structures."""
    print(f"=== [Phase 1] Pre-training Controller on N={pretrain_n} ===")
    
    pretrain_history = []
    for epoch in range(1, pretrain_epochs + 1):
        batch_log_probs, batch_rewards = [], []
        
        for _ in range(batch_size):
            program, log_prob = controller.sample_program()
            reward, _, _ = evaluate_program(
                program, test_cases=10, array_len=pretrain_n, vm_max_steps=500
            )
            batch_log_probs.append(log_prob)
            batch_rewards.append(reward)
            
        baseline = np.quantile(batch_rewards, quantile)
        policy_loss = sum([-lp * (r - baseline) for lp, r in zip(batch_log_probs, batch_rewards)]) / batch_size
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        avg_reward = float(np.mean(batch_rewards))
        pretrain_history.append(avg_reward)
        
        if epoch % 10 == 0 or epoch == pretrain_epochs:
            print(f"  Pre-train Epoch {epoch:02d}/{pretrain_epochs} | Avg Reward: {avg_reward:.3f}")
            
    print("=== [Phase 1] Pre-training Complete! Transferring Weights... ===\n")
    return controller, pretrain_history


def train_and_evaluate_dso_n20(
    pretrain=True,
    pretrain_epochs=40,
    fine_tune_epochs=80,
    batch_size_n20=64,
    lr=0.002,
    quantile_n20=0.85,
    save_dir="./dso_n20_results"
):
    os.makedirs(save_dir, exist_ok=True)
    controller = DSOController(VOCAB_SIZE)
    optimizer = optim.Adam(controller.parameters(), lr=lr)
    
    pretrain_history = []
    if pretrain:
        controller, pretrain_history = pretrain_controller(
            controller, optimizer, pretrain_epochs=pretrain_epochs, batch_size=32, pretrain_n=5
        )

    print(f"=== [Phase 2] Fine-tuning Controller on Target N=20 ===")
    
    history = {"epoch": [], "max_reward": [], "mean_acc": [], "mean_fid": []}
    best_program = None
    best_reward = -float("inf")
    
    for epoch in range(1, fine_tune_epochs + 1):
        batch_log_probs, batch_rewards, batch_accs, batch_fids = [], [], [], []
        
        for _ in range(batch_size_n20):
            program, log_prob = controller.sample_program()
            reward, acc, fid = evaluate_program(
                program, test_cases=25, array_len=20, vm_max_steps=2500
            )
            
            batch_log_probs.append(log_prob)
            batch_rewards.append(reward)
            batch_accs.append(acc)
            batch_fids.append(fid)
            
            if reward > best_reward:
                best_reward = reward
                best_program = program

        baseline = np.quantile(batch_rewards, quantile_n20)
        policy_loss = sum([-lp * (r - baseline) for lp, r in zip(batch_log_probs, batch_rewards)]) / batch_size_n20
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        history["epoch"].append(epoch)
        history["max_reward"].append(float(np.max(batch_rewards)))
        history["mean_acc"].append(float(np.mean(batch_accs)))
        history["mean_fid"].append(float(np.mean(batch_fids)))
        
        if epoch % 10 == 0 or epoch == fine_tune_epochs:
            print(f"  N=20 Epoch {epoch:02d}/{fine_tune_epochs} | "
                  f"Max Reward: {np.max(batch_rewards):.3f} | "
                  f"Avg Acc: {np.mean(batch_accs)*100:.1f}% | "
                  f"Avg Fid: {np.mean(batch_fids)*100:.1f}%")

    # Out-of-sample Testing on N=20
    test_reward, test_acc, test_fid = evaluate_program(
        best_program, test_cases=50, array_len=20, vm_max_steps=2500
    )
    
    print("\n================ FINAL TEST RESULTS (N=20) ================")
    print(f" Sorting Success Rate: {test_acc * 100:.2f}%")
    print(f" Trace Fidelity:       {test_fid * 100:.2f}%")
    print(f" Synthesized AST:      {[t.value for t in best_program if t != TokenType.STOP]}")
    print("===========================================================")

    return history, test_acc, test_fid


if __name__ == "__main__":
    train_and_evaluate_dso_n20(
        pretrain=True,
        pretrain_epochs=40,
        fine_tune_epochs=160,
        batch_size_n20=64,
        lr=0.002,
        quantile_n20=0.85
    )
