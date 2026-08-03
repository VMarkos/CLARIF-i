import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import generic DSO components
from dso_generic import (
    IDX_TO_TOKEN,
    TOKEN_TO_IDX,
    VOCAB_SIZE,
    AlgorithmCoach,
    DSOController,
    GenericVirtualMachine,
    TokenType,
    get_valid_mask,
    levenshtein_distance,
)

# =====================================================================
# 1. AST Extraction from Loaded Controller
# =====================================================================

def extract_best_program(controller, max_length=12):
    """Extracts the program AST greedily (argmax logits) using grammar masking."""
    controller.eval()
    tokens = []

    hx = torch.zeros(1, controller.hidden_dim)
    cx = torch.zeros(1, controller.hidden_dim)

    # Generic start token
    input_tok = torch.tensor([TOKEN_TO_IDX[TokenType.FOR_I_0_N]])
    tokens.append(IDX_TO_TOKEN[input_tok.item()])

    with torch.no_grad():
        for _ in range(max_length - 1):
            embed = controller.embedding(input_tok)
            hx, cx = controller.lstm(embed, (hx, cx))
            logits = controller.fc(hx).squeeze(0)

            # Apply constrained grammar mask to ensure valid greedy steps
            mask = get_valid_mask(tokens)
            logits[~mask] = -1e9

            # Greedy selection
            action = torch.argmax(logits, dim=-1)
            token = IDX_TO_TOKEN[action.item()]
            tokens.append(token)

            if token == TokenType.STOP:
                break
            input_tok = action.unsqueeze(0)

    return tokens

# =====================================================================
# 2. Pretty-Printing Module for Generic DSL
# =====================================================================

def pretty_print_ast(program_tokens, title="Synthesized Generic Sorting Algorithm"):
    """Parses generic DSL tokens and prints them as indented pseudocode."""
    indent_level = 0
    indent_str = "    "
    
    token_format = {
        TokenType.FOR_I_0_N: "for i = 0 to N-1:",
        TokenType.FOR_J_0_N_I_1: "for j = 0 to N - i - 2:",
        TokenType.FOR_J_I1_N: "for j = i + 1 to N - 1:",
        TokenType.FOR_J_I_0_DEC: "for j = i downto 1:",
        TokenType.IF_A_J_GT_A_J_PLUS1: "if A[j] > A[j+1]:",
        TokenType.IF_A_J_LT_A_MIN: "if A[j] < A[min_idx]:",
        TokenType.IF_A_J_LT_A_J_MIN1: "if A[j] < A[j-1]:",
        TokenType.SET_MIN_I: "min_idx = i",
        TokenType.SET_MIN_J: "min_idx = j",
        TokenType.SWAP_J_J_PLUS1: "swap(A[j], A[j+1])",
        TokenType.SWAP_I_MIN: "swap(A[i], A[min_idx])",
        TokenType.SWAP_J_J_MIN1: "swap(A[j], A[j-1])",
        TokenType.STOP: "<STOP>"
    }
    
    active_tokens = [t for t in program_tokens if t != TokenType.STOP]
    
    print("=" * 55)
    print(f" {title.upper()}")
    print("=" * 55)
    print("def synthesized_sort(A, N):")
    indent_level += 1

    for token in active_tokens:
        line = token_format.get(token, str(token.value))
        print(f"{indent_str * indent_level}{line}")
        
        if "for " in line or "if " in line:
            indent_level += 1

    print("=" * 55 + "\n")

# =====================================================================
# 3. Benchmark Evaluation Routine
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

    # Inspect checkpoint weights to dynamically set controller vocab size
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    checkpoint_vocab_size = state_dict['embedding.weight'].shape[0]

    controller = DSOController(checkpoint_vocab_size)
    controller.load_state_dict(state_dict)
    print(f"Successfully loaded checkpoint from: {model_path} (Vocab Size: {checkpoint_vocab_size})")

    # Extract synthesized program AST
    program_tokens = extract_best_program(controller)
    ast_str = [t.value for t in program_tokens if t != TokenType.STOP]
    
    # Pretty print the extracted AST
    pretty_print_ast(program_tokens, title=f"Extracted Program ({coach_type.title()} Coach Model)")

    results = {}
    vm = GenericVirtualMachine(max_op_steps=3000)

    # Select Coach Function
    coach_func = {
        "bubble": AlgorithmCoach.bubble_sort,
        "selection": AlgorithmCoach.selection_sort,
        "insertion": AlgorithmCoach.insertion_sort,
    }.get(coach_type.lower(), AlgorithmCoach.bubble_sort)

    print(f"--- Running Out-of-Sample Benchmark against {coach_type.upper()} Coach ---")
    for n in test_n_values:
        successes = 0
        fidelities = []

        for _ in range(num_test_cases):
            arr = np.random.permutation(n).tolist()
            _, coach_trace = coach_func(arr)

            # Execute loaded program on Virtual Machine
            _, prog_trace, is_valid_sort = vm.execute(program_tokens, arr)

            if is_valid_sort:
                successes += 1

            dist = levenshtein_distance(prog_trace, coach_trace)
            max_len = max(len(prog_trace), len(coach_trace), 1)
            fidelity = 1.0 - (dist / max_len)
            fidelities.append(fidelity)

        accuracy = (successes / num_test_cases) * 100
        mean_fidelity = np.mean(fidelities) * 100

        results[n] = {"accuracy": accuracy, "fidelity": mean_fidelity}
        print(
            f"  N={n:02d} | Sorting Accuracy: {accuracy:6.2f}% | Coach Fidelity: {mean_fidelity:6.2f}%"
        )

    return results, ast_str

# =====================================================================
# 4. Plotting Utilities
# =====================================================================

def plot_evaluation_results(results, coach_type="bubble", save_plot_path="evaluation_plot.png"):
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
        label=f"{coach_type.title()} Coach Fidelity (%)",
        color="tab:purple",
    )

    ax.set_xlabel("Array Size (N)", fontsize=11)
    ax.set_ylabel("Performance (%)", fontsize=11)
    ax.set_title(
        f"Generic DSO Model Testing Across Array Sizes ({coach_type.title()} Coach)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in n_values])
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper right")

    ax.bar_label(rects1, fmt="%.0f%%", padding=3, fontsize=9)
    ax.bar_label(rects2, fmt="%.0f%%", padding=3, fontsize=9)

    fig.tight_layout()
    plt.savefig(save_plot_path, dpi=300)
    print(f"\nEvaluation plot successfully saved to: {save_plot_path}")
    plt.show()

# =====================================================================
# 5. Main Execution
# =====================================================================

if __name__ == "__main__":
    # Choose which trained model to evaluate: 'bubble', 'insertion', or 'selection'
    COACH_TYPE = "bubble"
    CHECKPOINT_PATH = f"./dso_results/controller_{COACH_TYPE}.pt"

    test_results, ast = evaluate_loaded_model(
        model_path=CHECKPOINT_PATH,
        test_n_values=[3, 5, 10, 15, 20],
        num_test_cases=100,
        coach_type=COACH_TYPE,
    )

    plot_evaluation_results(
        test_results,
        coach_type=COACH_TYPE,
        save_plot_path=f"dso_model_test_metrics_{COACH_TYPE}.png"
    )
