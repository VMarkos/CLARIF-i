import enum
import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# =====================================================================
# 1. Parameterized DSL Definitions
# =====================================================================


class TokenType(enum.Enum):
  # Loop Primitives
  FOR_I_0_N = "FOR i IN 0..N"
  FOR_J_0_N_I_1 = "FOR j IN 0..N-i-1"  # Bubble
  FOR_J_I1_N = "FOR j IN (i+1)..N"  # Selection
  FOR_J_I_0_DEC = "FOR j IN i DOWNTO 1"  # Insertion

  # Comparison Primitives
  IF_A_J_GT_A_J_PLUS1 = "IF A[j] > A[j+1]"  # Bubble
  IF_A_J_LT_A_MIN = "IF A[j] < A[min_idx]"  # Selection
  IF_A_J_LT_A_J_MIN1 = "IF A[j] < A[j-1]"  # Insertion

  # Pointer / State Operations
  SET_MIN_I = "min_idx = i"
  SET_MIN_J = "min_idx = j"

  # Swap Operations
  SWAP_J_J_PLUS1 = "SWAP(A[j], A[j+1])"  # Bubble
  SWAP_I_MIN = "SWAP(A[i], A[min_idx])"  # Selection
  SWAP_J_J_MIN1 = "SWAP(A[j], A[j-1])"  # Insertion

  STOP = "<STOP>"


DSL_TOKENS = list(TokenType)
TOKEN_TO_IDX = {token: idx for idx, token in enumerate(DSL_TOKENS)}
IDX_TO_TOKEN = {idx: token for idx, token in enumerate(DSL_TOKENS)}
VOCAB_SIZE = len(DSL_TOKENS)

# =====================================================================
# 2. Reference Algorithm Coaches & Levenshtein Distance
# =====================================================================


class AlgorithmCoach:

  @staticmethod
  def bubble_sort(arr):
    arr, trace, n = list(arr), [], len(arr)
    for i in range(n):
      for j in range(0, n - i - 1):
        trace.append(("COMPARE", j, j + 1))
        if arr[j] > arr[j + 1]:
          arr[j], arr[j + 1] = arr[j + 1], arr[j]
          trace.append(("SWAP", j, j + 1))
    return arr, trace

  @staticmethod
  def selection_sort(arr):
    arr, trace, n = list(arr), [], len(arr)
    for i in range(n):
      min_idx = i
      for j in range(i + 1, n):
        trace.append(("COMPARE", j, min_idx))
        if arr[j] < arr[min_idx]:
          min_idx = j
      if min_idx != i:
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        trace.append(("SWAP", i, min_idx))
    return arr, trace

  @staticmethod
  def insertion_sort(arr):
    arr, trace, n = list(arr), [], len(arr)
    for i in range(1, n):
      j = i
      while j > 0:
        trace.append(("COMPARE", j, j - 1))
        if arr[j] < arr[j - 1]:
          arr[j], arr[j - 1] = arr[j - 1], arr[j]
          trace.append(("SWAP", j, j - 1))
          j -= 1
        else:
          break
    return arr, trace


def levenshtein_distance(seq1, seq2):
  m, n = len(seq1), len(seq2)
  dp = [[0] * (n + 1) for _ in range(m + 1)]
  for i in range(m + 1):
    dp[i][0] = i
  for j in range(n + 1):
    dp[0][j] = j
  for i in range(1, m + 1):
    for j in range(1, n + 1):
      if seq1[i - 1] == seq2[j - 1]:
        dp[i][j] = dp[i - 1][j - 1]
      else:
        dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
  return dp[m][n]


# =====================================================================
# 3. Constrained Grammar Masking
# =====================================================================


def get_valid_mask(current_tokens):
  mask = torch.ones(VOCAB_SIZE, dtype=torch.bool)
  if not current_tokens:
    mask[:] = False
    mask[TOKEN_TO_IDX[TokenType.FOR_I_0_N]] = True
    return mask

  last_token = current_tokens[-1]

  if last_token == TokenType.FOR_I_0_N:
    mask[:] = False
    mask[TOKEN_TO_IDX[TokenType.FOR_J_0_N_I_1]] = True
    mask[TOKEN_TO_IDX[TokenType.FOR_J_I1_N]] = True
    mask[TOKEN_TO_IDX[TokenType.FOR_J_I_0_DEC]] = True
    mask[TOKEN_TO_IDX[TokenType.SET_MIN_I]] = True

  elif last_token == TokenType.SET_MIN_I:
    mask[:] = False
    mask[TOKEN_TO_IDX[TokenType.FOR_J_I1_N]] = True

  elif last_token == TokenType.FOR_J_0_N_I_1:
    mask[:] = False
    mask[TOKEN_TO_IDX[TokenType.IF_A_J_GT_A_J_PLUS1]] = True

  elif last_token == TokenType.FOR_J_I1_N:
    mask[:] = False
    mask[TOKEN_TO_IDX[TokenType.IF_A_J_LT_A_MIN]] = True

  elif last_token == TokenType.FOR_J_I_0_DEC:
    mask[:] = False
    mask[TOKEN_TO_IDX[TokenType.IF_A_J_LT_A_J_MIN1]] = True

  elif last_token == TokenType.IF_A_J_GT_A_J_PLUS1:
    mask[:] = False
    mask[TOKEN_TO_IDX[TokenType.SWAP_J_J_PLUS1]] = True

  elif last_token == TokenType.IF_A_J_LT_A_J_MIN1:
    mask[:] = False
    mask[TOKEN_TO_IDX[TokenType.SWAP_J_J_MIN1]] = True

  elif last_token == TokenType.IF_A_J_LT_A_MIN:
    mask[:] = False
    mask[TOKEN_TO_IDX[TokenType.SET_MIN_J]] = True

  elif last_token == TokenType.SET_MIN_J:
    mask[:] = False
    mask[TOKEN_TO_IDX[TokenType.SWAP_I_MIN]] = True

  elif last_token in (
      TokenType.SWAP_J_J_PLUS1,
      TokenType.SWAP_J_J_MIN1,
      TokenType.SWAP_I_MIN,
  ):
    mask[:] = False
    mask[TOKEN_TO_IDX[TokenType.STOP]] = True

  return mask


# =====================================================================
# 4. Generic Virtual Machine Execution Engine
# =====================================================================


class GenericVirtualMachine:

  def __init__(self, max_op_steps=3000):
    self.max_op_steps = max_op_steps

  def execute(self, program_tokens, input_array):
    arr = list(input_array)
    n = len(arr)
    trace = []
    op_count = 0
    min_idx = 0

    has_set_min_i = TokenType.SET_MIN_I in program_tokens
    has_set_min_j = TokenType.SET_MIN_J in program_tokens
    has_swap_i_min = TokenType.SWAP_I_MIN in program_tokens

    try:
      if TokenType.FOR_I_0_N in program_tokens:
        for i in range(n):
          if has_set_min_i:
            min_idx = i

          # Bubble Branch
          if TokenType.FOR_J_0_N_I_1 in program_tokens:
            for j in range(0, max(0, n - i - 1)):
              op_count += 1
              if op_count > self.max_op_steps:
                break
              if TokenType.IF_A_J_GT_A_J_PLUS1 in program_tokens:
                trace.append(("COMPARE", j, j + 1))
                if arr[j] > arr[j + 1]:
                  if TokenType.SWAP_J_J_PLUS1 in program_tokens:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    trace.append(("SWAP", j, j + 1))

          # Selection Branch
          elif TokenType.FOR_J_I1_N in program_tokens:
            for j in range(i + 1, n):
              op_count += 1
              if op_count > self.max_op_steps:
                break
              if TokenType.IF_A_J_LT_A_MIN in program_tokens:
                trace.append(("COMPARE", j, min_idx))
                if arr[j] < arr[min_idx]:
                  if has_set_min_j:
                    min_idx = j

            if has_swap_i_min and min_idx != i:
              arr[i], arr[min_idx] = arr[min_idx], arr[i]
              trace.append(("SWAP", i, min_idx))

          # Insertion Branch
          elif TokenType.FOR_J_I_0_DEC in program_tokens:
            for j in range(i, 0, -1):
              op_count += 1
              if op_count > self.max_op_steps:
                break
              if TokenType.IF_A_J_LT_A_J_MIN1 in program_tokens:
                trace.append(("COMPARE", j, j - 1))
                if arr[j] < arr[j - 1]:
                  if TokenType.SWAP_J_J_MIN1 in program_tokens:
                    arr[j], arr[j - 1] = arr[j - 1], arr[j]
                    trace.append(("SWAP", j, j - 1))
                else:
                  break

          if op_count > self.max_op_steps:
            break

    except Exception:
      pass

    has_swaps = any(evt[0] == "SWAP" for evt in trace)
    is_sorted = arr == sorted(input_array)
    valid_execution = is_sorted and (
        has_swaps or input_array == sorted(input_array)
    )
    return arr, trace, valid_execution


# =====================================================================
# 5. Controller Network
# =====================================================================


class DSOController(nn.Module):

  def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
    self.fc = nn.Linear(hidden_dim, vocab_size)
    self.hidden_dim = hidden_dim

  def sample_program(self, max_length=12):
    tokens, log_probs = [], []
    hx = torch.zeros(1, self.hidden_dim)
    cx = torch.zeros(1, self.hidden_dim)

    input_tok = torch.tensor([TOKEN_TO_IDX[TokenType.FOR_I_0_N]])
    tokens.append(IDX_TO_TOKEN[input_tok.item()])

    for _ in range(max_length - 1):
      embed = self.embedding(input_tok)
      hx, cx = self.lstm(embed, (hx, cx))
      logits = self.fc(hx).squeeze(0)

      mask = get_valid_mask(tokens)
      logits[~mask] = -1e9

      dist = Categorical(logits=logits)
      action = dist.sample()

      log_probs.append(dist.log_prob(action))
      token = IDX_TO_TOKEN[action.item()]
      tokens.append(token)

      if token == TokenType.STOP:
        break
      input_tok = action.unsqueeze(0)

    return tokens, torch.stack(log_probs).sum()


# =====================================================================
# 6. Evaluation Routine with Coach Bias
# =====================================================================


def evaluate_program(
    program_tokens,
    coach_type="bubble",
    test_cases=20,
    array_len=20,
    vm_max_steps=3000,
):
  vm = GenericVirtualMachine(max_op_steps=vm_max_steps)
  total_success = 0
  fidelities = []

  coach_func = {
      "bubble": AlgorithmCoach.bubble_sort,
      "selection": AlgorithmCoach.selection_sort,
      "insertion": AlgorithmCoach.insertion_sort,
  }.get(coach_type.lower(), AlgorithmCoach.bubble_sort)

  for _ in range(test_cases):
    arr = np.random.permutation(array_len).tolist()
    _, coach_trace = coach_func(arr)
    _, prog_trace, is_valid_sort = vm.execute(program_tokens, arr)

    if is_valid_sort:
      total_success += 1

    dist = levenshtein_distance(prog_trace, coach_trace)
    max_len = max(len(prog_trace), len(coach_trace), 1)
    fidelity = 1.0 - (dist / max_len)
    fidelities.append(fidelity)

  accuracy = total_success / test_cases
  mean_fidelity = float(np.mean(fidelities))
  reward = 0.5 * accuracy + 0.5 * mean_fidelity
  return reward, accuracy, mean_fidelity


# =====================================================================
# 7. Training Pipeline with Saving & Plotting Utilities
# =====================================================================


def train_generic_dso(
    coach_type="bubble",
    pretrain_epochs=30,
    fine_tune_epochs=50,
    batch_size=64,
    lr=0.002,
    quantile=0.85,
    target_len=20,
    save_dir="./dso_generic_results",
):

  os.makedirs(save_dir, exist_ok=True)
  controller = DSOController(VOCAB_SIZE)
  optimizer = optim.Adam(controller.parameters(), lr=lr)

  history = {"epoch": [], "max_reward": [], "mean_acc": [], "mean_fid": []}

  # --- Phase 1: Pre-training (N=5) ---
  print(
      f"=== [Phase 1] Pre-training Controller (N=5) for Coach:"
      f" {coach_type.upper()} ==="
  )
  for epoch in range(1, pretrain_epochs + 1):
    batch_log_probs, batch_rewards = [], []
    for _ in range(32):
      program, log_prob = controller.sample_program()
      r, _, _ = evaluate_program(
          program,
          coach_type=coach_type,
          test_cases=10,
          array_len=5,
          vm_max_steps=500,
      )
      batch_log_probs.append(log_prob)
      batch_rewards.append(r)

    baseline = np.quantile(batch_rewards, 0.75)
    loss = (
        sum([-lp * (r - baseline) for lp, r in zip(batch_log_probs, batch_rewards)])
        / 32
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == pretrain_epochs:
      print(
          f"  Pre-train Epoch {epoch:02d}/{pretrain_epochs} | Avg Reward:"
          f" {np.mean(batch_rewards):.3f}"
      )

  # --- Phase 2: Fine-tuning on N=20 ---
  print(
      f"\n=== [Phase 2] Target Training (N=20) for Coach:"
      f" {coach_type.upper()} ==="
  )
  best_reward, best_program = -float("inf"), None

  for epoch in range(1, fine_tune_epochs + 1):
    batch_log_probs, batch_rewards, batch_accs, batch_fids = [], [], [], []

    for _ in range(batch_size):
      program, log_prob = controller.sample_program()
      r, acc, fid = evaluate_program(
          program,
          coach_type=coach_type,
          test_cases=20,
          array_len=target_len,
          vm_max_steps=3000,
      )

      batch_log_probs.append(log_prob)
      batch_rewards.append(r)
      batch_accs.append(acc)
      batch_fids.append(fid)

      if r > best_reward:
        best_reward = r
        best_program = program

    baseline = np.quantile(batch_rewards, quantile)
    loss = (
        sum(
            [-lp * (r - baseline) for lp, r in zip(batch_log_probs, batch_rewards)]
        )
        / batch_size
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    history["epoch"].append(epoch)
    history["max_reward"].append(float(np.max(batch_rewards)))
    history["mean_acc"].append(float(np.mean(batch_accs)))
    history["mean_fid"].append(float(np.mean(batch_fids)))

    if epoch % 10 == 0 or epoch == fine_tune_epochs:
      print(
          f"  Epoch {epoch:02d}/{fine_tune_epochs} | Max Reward:"
          f" {np.max(batch_rewards):.3f} | Acc: {np.mean(batch_accs)*100:.1f}% |"
          f" Fid: {np.mean(batch_fids)*100:.1f}%"
      )

  # --- Save Model Artifacts ---
  model_path = os.path.join(save_dir, f"controller_{coach_type}.pt")
  json_path = os.path.join(save_dir, f"history_{coach_type}.json")

  torch.save(controller.state_dict(), model_path)
  with open(json_path, "w") as f:
    json.dump(history, f, indent=2)

  print(f"\n[Saved] Checkpoint -> {model_path}")
  print(f"[Saved] Metrics -> {json_path}")

  # Pretty-print top synthesized AST
  print("\n================ SYNTHESIZED AST ================")
  for token in best_program:
    if token != TokenType.STOP:
      print(f"  -> {token.value}")
  print("=================================================")

  plot_training_curves(history, coach_type, save_dir)
  return controller, history


def plot_training_curves(history, coach_type, save_dir):
  """Plots convergence metrics across epochs."""
  plt.figure(figsize=(9, 5))
  epochs = history["epoch"]

  plt.plot(
      epochs,
      history["max_reward"],
      label="Max Reward",
      color="tab:blue",
      linewidth=2,
  )
  plt.plot(
      epochs,
      history["mean_acc"],
      label="Mean Accuracy",
      color="tab:orange",
      linestyle="--",
  )
  plt.plot(
      epochs,
      history["mean_fid"],
      label="Mean Coach Fidelity",
      color="tab:green",
      linestyle=":",
  )

  plt.title(
      f"DSO Convergence Curves (Coach: {coach_type.title()})",
      fontsize=12,
      fontweight="bold",
  )
  plt.xlabel("Fine-tuning Epochs (N=20)", fontsize=10)
  plt.ylabel("Score / Percentage", fontsize=10)
  plt.ylim(-0.05, 1.05)
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend(loc="lower right")

  plot_path = os.path.join(save_dir, f"training_curve_{coach_type}.png")
  plt.savefig(plot_path, dpi=300, bbox_inches="tight")
  plt.close()
  print(f"[Saved] Plot -> {plot_path}")


# =====================================================================
# 8. Main Entry Point
# =====================================================================

if __name__ == "__main__":
  # Change coach_type to 'bubble', 'selection', or 'insertion'
  TARGET_COACH = "bubble"

  train_generic_dso(
      coach_type=TARGET_COACH,
      pretrain_epochs=30,
      fine_tune_epochs=50,
      batch_size=64,
      lr=0.002,
      quantile=0.85,
      save_dir="./dso_results",
  )
