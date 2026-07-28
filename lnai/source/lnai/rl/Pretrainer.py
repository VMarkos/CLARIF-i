# Pretrainer.py
#
# Pretrainer utils

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy


# ---------------------------------------------------------------------------
# 1. Expert Action Functions (Algorithmic Oracles)
# ---------------------------------------------------------------------------

def bubble_sort_expert_action(arr: np.ndarray) -> int:
    """
    Given an array state, returns the next adjacent swap index (0 to len(arr)-2)
    prescribed by standard Bubble Sort.
    """
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                return j  # Execute swap(j, j+1)
    return -1  # Array is already sorted


def insertion_sort_expert_action(arr: np.ndarray) -> int:
    """
    Given an array state, returns the next adjacent swap index prescribed
    by Insertion Sort (shifting element leftward into correct position).
    """
    n = len(arr)
    for i in range(1, n):
        j = i
        while j > 0 and arr[j - 1] > arr[j]:
            return j - 1  # Execute swap(j-1, j) to slide element left
    return -1  # Array is already sorted


EXPERT_REGISTRY = {
    "bubble": bubble_sort_expert_action,
    "insertion": insertion_sort_expert_action,
}


# ---------------------------------------------------------------------------
# 2. DAgger Pre-training Routine
# ---------------------------------------------------------------------------

def pretrain_dagger(
    env,
    algorithm_name: str = "bubble",
    n_range: range = range(3, 6),
    samples_per_n: dict | None = None,
    dagger_iterations: int = 5,
    bc_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu"
) -> MaskablePPO:
    """
    Pre-trains a MaskablePPO agent using DAgger on expert algorithm trajectories.

    Args:
        env: CurriculumPathSearchEnv instance initialized with max_n.
        algorithm_name: 'bubble' or 'insertion'.
        n_range: Range of sizes n to train on (e.g., range(3, 6)).
        samples_per_n: Dict mapping n -> number of random arrays (e.g., {3: 50, 4: 100, 5: 200}).
        dagger_iterations: Number of interactive DAgger collection rounds.
        bc_epochs: Supervised training epochs per DAgger iteration.
        batch_size: DataLoader batch size.
        learning_rate: Optimizer learning rate for supervised BC phase.
        device: Torch device ('cpu' or 'cuda').

    Returns:
        model: Pre-trained MaskablePPO agent ready for RL fine-tuning.
    """
    if samples_per_n is None:
        samples_per_n = {n: 100 for n in n_range}

    expert_fn = EXPERT_REGISTRY[algorithm_name]

    # Instantiate the base PPO model with MultiInput policy
    model = MaskablePPO(
        MaskableMultiInputActorCriticPolicy,
        env,
        learning_rate=3e-4,
        verbose=0,
        device=device
    )
    
    # We directly update the policy network parameters via Supervised BC
    policy_net = model.policy.to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Datasets for DAgger: stores (observation_dict, expert_action)
    collected_obs_array = []
    collected_obs_mask = []
    collected_actions = []

    print(f"=== Starting DAgger Pre-training for '{algorithm_name.upper()}' ===")

    # -----------------------------------------------------------------------
    # Step A: Initial Expert Dataset Collection (Pure Expert Rollouts)
    # -----------------------------------------------------------------------
    for n in n_range:
        env.unwrapped.set_curriculum_level(n)
        num_samples = samples_per_n.get(n, 100)
        
        for _ in range(num_samples):
            obs, _ = env.reset()
            done = False
            
            while not done:
                current_arr = obs["current_array"][:n]
                expert_act = expert_fn(current_arr)
                
                if expert_act == -1:
                    break  # Already sorted

                # Store observation components and target expert action
                collected_obs_array.append(obs["current_array"])
                collected_obs_mask.append(obs["active_mask"])
                collected_actions.append(expert_act)

                # Step environment using expert action
                obs, reward, terminated, truncated, _ = env.step(expert_act)
                done = terminated or truncated

    print(f"Initial Expert Dataset: {len(collected_actions)} state-action pairs.")

    # -----------------------------------------------------------------------
    # Step B: DAgger Loop (Student Execution + Expert Correction)
    # -----------------------------------------------------------------------
    for dagger_iter in range(dagger_iterations):
        # 1. Supervised Behavioral Cloning Step
        tensor_arr = torch.tensor(np.array(collected_obs_array), dtype=torch.int32)
        tensor_mask = torch.tensor(np.array(collected_obs_mask), dtype=torch.int32)
        tensor_act = torch.tensor(np.array(collected_actions), dtype=torch.long)

        dataset = TensorDataset(tensor_arr, tensor_mask, tensor_act)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        policy_net.train()
        total_loss = 0.0
        for epoch in range(bc_epochs):
            for b_arr, b_mask, b_act in loader:
                b_arr, b_mask, b_act = b_arr.to(device), b_mask.to(device), b_act.to(device)

                # Format batch observation dict matching MultiInput Policy structure
                obs_dict = {
                    "current_array": b_arr,
                    "active_mask": b_mask
                }

                # Extract features and compute policy action logits
                features = policy_net.extract_features(obs_dict)
                latent_pi, _ = policy_net.mlp_extractor(features)
                logits = policy_net.action_net(latent_pi)

                loss = loss_fn(logits, b_act)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"DAgger Iter {dagger_iter + 1}/{dagger_iterations} | BC Loss: {avg_loss:.4f} | Dataset Size: {len(collected_actions)}")

        # 2. Collect New States using the STUDENT Policy, labeled by EXPERT
        if dagger_iter < dagger_iterations - 1:
            policy_net.eval()
            new_samples = 0
            
            for n in n_range:
                env.unwrapped.set_curriculum_level(n)
                for _ in range(20):  # Interactive rollouts per size n
                    obs, _ = env.reset()
                    done = False
                    
                    while not done:
                        # Get action mask & student action prediction
                        mask = env.action_masks()
                        with torch.no_grad():
                            student_action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                        
                        student_action = int(student_action.item()) if isinstance(student_action, np.ndarray) else int(student_action)

                        # Query EXPERT for correct action on this student-visited state
                        current_arr = obs["current_array"][:n]
                        expert_act = expert_fn(current_arr)

                        if expert_act != -1:
                            collected_obs_array.append(obs["current_array"])
                            collected_obs_mask.append(obs["active_mask"])
                            collected_actions.append(expert_act)
                            new_samples += 1

                        # Step environment with student action to explore student errors
                        obs, reward, terminated, truncated, _ = env.step(student_action)
                        done = terminated or truncated

    print("=== Pre-training Complete! ===")
    return model
