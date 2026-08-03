import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import components from your original DSO module
from dso_generic import (
    IDX_TO_TOKEN,
    TOKEN_TO_IDX,
    VOCAB_SIZE,
    AlgorithmCoach,
    DSOController,
    GenericVirtualMachine,
    TokenType,
    levenshtein_distance,
)

# =====================================================================
# 1. AST Extraction from Loaded Controller
# =====================================================================


def extract_best_program(controller, max_length=12):
  """Extracts the program AST greedily (argmax logits) from the loaded model."""
  controller.eval()
  tokens = []

  hx = torch.zeros(1, controller.hidden_dim)
  cx = torch.zeros(1, controller.hidden_dim)

  input_tok = torch.tensor([TOKEN_TO_IDX[TokenType.FOR_I]])
  tokens.append(IDX_TO_TOKEN[input_tok.item()])

  with torch.no_grad():
    for _ in range(max_length - 1):
      embed = controller.embedding(input_tok)
      hx, cx = controller.lstm(embed, (hx, cx))
      logits = controller.fc(hx)

      # Deterministic selection for evaluation
      action = torch.argmax(logits, dim=-1)
      token = IDX_TO_TOKEN[action.item()]
      tokens.append(token)

      if token == TokenType.STOP:
        break
      input_tok = action

  return tokens


# =====================================================================
# 2. Benchmark Evaluation Routine
# =====================================================================


def evaluate_loaded_model(
    model_path,
    test_n_values=[3, 5, 10, 15, 20],
    num_test_cases=50,
    coach_type="bubble",
):
  """Loads checkpoint, runs evaluation across array sizes N, and returns metrics."""
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

  # Load model weights
  controller = DSOController(VOCAB_SIZE)
  controller.load_state_dict(
      torch.load(model_path, map_location=torch.device("cpu"))
  )
  print(f"Successfully loaded checkpoint from: {model_path}")

  # Extract synthesized program AST
  program_tokens = extract_best_program(controller)
  ast_str = [t.value for t in program_tokens if t != TokenType.STOP]
  print(f"\nExtracted Program AST:\n  {ast_str}\n")

  results = {}
  vm = GenericVirtualMachine(max_op_steps=2500)

  print("--- Running Out-of-Sample Benchmark ---")
  for n in test_n_values:
    successes = 0
    fidelities = []

    for _ in range(num_test_cases):
      # Generate random permutation of length n
      arr = np.random.permutation(n).tolist()

      if coach_type == "bubble":
        _, coach_trace = AlgorithmCoach.bubble_sort(arr)
      else:
        raise NotImplementedError("Coach type not implemented.")

      # Execute loaded program on Virtual Machine
      _, prog_trace, is_sorted = vm.execute(program_tokens, arr)

      if is_sorted:
        successes += 1

      dist = levenshtein_distance(prog_trace, coach_trace)
      max_len = max(len(prog_trace), len(coach_trace), 1)
      fidelity = 1.0 - (dist / max_len)
      fidelities.append(fidelity)

    accuracy = (successes / num_test_cases) * 100
    mean_fidelity = np.mean(fidelities) * 100

    results[n] = {"accuracy": accuracy, "fidelity": mean_fidelity}
    print(
        f"  N={n:02d} | Sorting Accuracy: {accuracy:6.2f}% | Coach Fidelity:"
        f" {mean_fidelity:6.2f}%"
    )

  return results, ast_str


# =====================================================================
# 3. Plotting Utilities
# =====================================================================


def plot_evaluation_results(results, save_plot_path="evaluation_plot.png"):
  """Plots accuracy and fidelity metrics across array sizes N."""
  n_values = list(results.keys())
  accuracies = [results[n]["accuracy"] for n in n_values]
  fidelities = [results[n]["fidelity"] for n in n_values]

  x = np.arange(len(n_values))
  width = 0.35

  fig, ax = plt.subplots(figsize=(9, 5))

  rects1 = ax.bar(
      x - width / 2,
      accuracies,
      width,
      label="Sorting Accuracy (%)",
      color="tab:orange",
  )
  rects2 = ax.bar(
      x + width / 2,
      fidelities,
      width,
      label="Coach Fidelity (%)",
      color="tab:purple",
  )

  ax.set_xlabel("Array Size (N)", fontsize=11)
  ax.set_ylabel("Performance (%)", fontsize=11)
  ax.set_title(
      "DSO Model Testing Across Array Sizes", fontsize=13, fontweight="bold"
  )
  ax.set_xticks(x)
  ax.set_xticklabels([f"N={n}" for n in n_values])
  ax.set_ylim(0, 110)
  ax.grid(True, linestyle=":", alpha=0.6, axis="y")
  ax.legend(loc="upper right")

  # Add text labels on top of bars
  ax.bar_label(rects1, fmt="%.0f%%", padding=3, fontsize=9)
  ax.bar_label(rects2, fmt="%.0f%%", padding=3, fontsize=9)

  fig.tight_layout()
  plt.savefig(save_plot_path, dpi=300)
  print(f"\nEvaluation plot successfully saved to: {save_plot_path}")
  plt.show()


def pretty_print_ast(program_tokens, title="Synthesized Sorting Algorithm"):
    """
    Parses a list of DSL tokens and prints them as indented, structured pseudocode.
    """
    indent_level = 0
    indent_str = "    "  # 4 spaces per indentation level
    
    # Mapping tokens to human-readable pseudocode lines
    token_format = {
        TokenType.FOR_I: "for i = 0 to N-1:",
        TokenType.FOR_J_BUBBLE: "for j = 0 to N - i - 2:",
        TokenType.FOR_J_SELECTION: "for j = i + 1 to N - 1:",
        TokenType.IF_GT: "if A[j] > A[j+1]:",
        TokenType.IF_GT_MIN: "if A[j] < A[min_idx]:",
        TokenType.SWAP_ADJACENT: "swap(A[j], A[j+1])",
        TokenType.SWAP_I_MIN: "swap(A[i], A[min_idx])",
        TokenType.SET_MIN_J: "min_idx = j",
        TokenType.END_BLOCK: "end",
        TokenType.STOP: "<STOP>"
    }
    
    # Filtering out STOP tokens
    active_tokens = [t for t in program_tokens if t != TokenType.STOP]
    
    print("=" * 50)
    print(f" {title.upper()}")
    print("=" * 50)
    print("def synthesized_sort(A, N):")
    indent_level += 1

    for token in active_tokens:
        line = token_format.get(token, str(token.value))
        
        # Adjust indentation level before printing END blocks
        if token == TokenType.END_BLOCK:
            indent_level = max(0, indent_level - 1)
            
        # Print formatted line with current indentation
        print(f"{indent_str * indent_level}{line}")
        
        # Increase indentation for control flow statements
        if "for " in line or "if " in line:
            indent_level += 1

    print("=" * 50 + "\n")



# =====================================================================
# 4. Main Execution
# =====================================================================

if __name__ == "__main__":
  # Specify path to your saved .pt model checkpoint
  CHECKPOINT_PATH = "./dso_results/controller_bubble.pt"

  test_results, ast = evaluate_loaded_model(
      model_path=CHECKPOINT_PATH,
      test_n_values=[3, 5, 10, 15, 20],
      num_test_cases=100,
      coach_type="bubble",
  )

  controller = DSOController(VOCAB_SIZE)
  controller.load_state_dict(torch.load(CHECKPOINT_PATH))

  program_tokens = extract_best_program(controller)

  pretty_print_ast(program_tokens, title='Best Program!')

  plot_evaluation_results(
      test_results, save_plot_path="dso_model_test_metrics.png"
  )
