import itertools
import math
import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback

# ==========================================
# 1. Custom Gymnasium Environment
# ==========================================


class SwapSortingEnv(gym.Env):

  def __init__(self, max_n=10, initial_n=3, max_steps=50):
    super().__init__()
    self.max_n = max_n
    self.current_n = initial_n
    self.max_steps = max_steps
    self.steps_taken = 0

    # Observation space: array containing values 1 to max_n
    self.observation_space = spaces.Box(
        low=1, high=self.max_n, shape=(self.max_n,), dtype=np.int32
    )

    # Precompute mapping for C(max_n, 2) unique swap pairs (i, j) where i < j
    self.swap_pairs = list(itertools.combinations(range(self.max_n), 2))
    self.num_actions = len(self.swap_pairs)
    self.action_space = spaces.Discrete(self.num_actions)

    self.state = None
    self.prev_inversions = 0

  def set_n(self, n):
    """Update current curriculum stage n."""
    self.current_n = min(n, self.max_n)

  def action_masks(self):
    """Action mask for MaskablePPO: allows only swaps with indices < current_n."""
    mask = np.zeros(self.num_actions, dtype=bool)
    for idx, (i, j) in enumerate(self.swap_pairs):
      if i < self.current_n and j < self.current_n:
        mask[idx] = True
    return mask

  def _count_inversions(self, arr):
    """Count inversions in the active slice [0 : current_n]."""
    slice_arr = arr[: self.current_n]
    inv = 0
    for i in range(len(slice_arr)):
      for j in range(i + 1, len(slice_arr)):
        if slice_arr[i] > slice_arr[j]:
          inv += 1
    return inv

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.steps_taken = 0

    # Initialize array: active part [0:current_n] shuffled, rest statically sorted
    active_slice = np.arange(1, self.current_n + 1, dtype=np.int32)
    # Ensure starting state is not already sorted
    while True:
      self.np_random.shuffle(active_slice)
      if self._count_inversions(active_slice) > 0 or self.current_n <= 1:
        break

    self.state = np.arange(1, self.max_n + 1, dtype=np.int32)
    self.state[: self.current_n] = active_slice
    self.prev_inversions = self._count_inversions(self.state)

    return self.state.copy(), {}

  def step(self, action):
    self.steps_taken += 1
    i, j = self.swap_pairs[action]

    # Perform swap
    self.state[i], self.state[j] = self.state[j], self.state[i]

    current_inversions = self._count_inversions(self.state)

    # Calculate potential-based reward: delta in inversions
    delta_inv = self.prev_inversions - current_inversions
    self.prev_inversions = current_inversions

    reward = float(delta_inv) * 2.0 - 0.1  # Step living penalty

    terminated = current_inversions == 0
    if terminated:
      reward += 10.0  # Completion bonus

    truncated = self.steps_taken >= self.max_steps

    return self.state.copy(), reward, terminated, truncated, {}


# ==========================================
# 2. SB3 Curriculum Callback
# ==========================================


class CurriculumCallback(BaseCallback):

  def __init__(
      self,
      eval_env,
      target_success_rate=0.85,
      eval_freq=2000,
      n_eval_episodes=20,
      verbose=1,
  ):
    super().__init__(verbose)
    self.eval_env = eval_env
    self.target_success_rate = target_success_rate
    self.eval_freq = eval_freq
    self.n_eval_episodes = n_eval_episodes
    self.current_n = eval_env.current_n

  def _on_step(self) -> bool:
    if self.n_calls % self.eval_freq == 0:
      successes = 0
      for _ in range(self.n_eval_episodes):
        obs, _ = self.eval_env.reset()
        done = False
        while not done:
          # Query mask for evaluation model prediction
          action_masks = self.eval_env.action_masks()
          action, _ = self.model.predict(
              obs, action_masks=action_masks, deterministic=True
          )
          obs, reward, terminated, truncated, _ = self.eval_env.step(action)
          done = terminated or truncated
          if terminated:
            successes += 1

      success_rate = successes / self.n_eval_episodes
      if self.verbose > 0:
        print(
            f"[Step {self.n_calls}] Stage n={self.current_n} | Success Rate:"
            f" {success_rate:.2f}"
        )

      if (
          success_rate >= self.target_success_rate
          and self.current_n < self.eval_env.max_n
      ):
        self.current_n += 1
        self.max_steps = max(40, self.current_n ** 2)
        if self.verbose > 0:
          print(f"--> Advancing Curriculum Stage to n = {self.current_n}!")

        # Update both training and evaluation envs using SB3 env_method
        self.training_env.env_method("set_n", self.current_n)
        self.eval_env.set_n(self.current_n)

    return True


# ==========================================
# 3. Evaluation Routine & Plotting
# ==========================================


def evaluate_agent(model, env, eval_stages, num_episodes=50):
  """Evaluates trained agent across different values of n."""
  results = {}

  for n in eval_stages:
    env.set_n(n)
    successes = 0
    step_counts = []

    for _ in range(num_episodes):
      obs, _ = env.reset()
      done = False
      steps = 0

      while not done:
        action_masks = env.action_masks()
        action, _ = model.predict(
            obs, action_masks=action_masks, deterministic=True
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1

        if terminated:
          successes += 1
          step_counts.append(steps)

    success_rate = (successes / num_episodes) * 100
    avg_steps = np.mean(step_counts) if step_counts else env.max_steps
    results[n] = {"success_rate": success_rate, "avg_steps": avg_steps}

  return results


def plot_results(results):
  """Plots evaluation success rate and average steps across n."""
  n_values = list(results.keys())
  success_rates = [results[n]["success_rate"] for n in n_values]
  avg_steps = [results[n]["avg_steps"] for n in n_values]

  fig, ax1 = plt.subplots(figsize=(8, 4.5))

  color = "tab:blue"
  ax1.set_xlabel("List Size (n)", fontsize=12)
  ax1.set_ylabel("Success Rate (%)", color=color, fontsize=12)
  ax1.plot(
      n_values,
      success_rates,
      color=color,
      marker="o",
      linewidth=2,
      label="Success Rate",
  )
  ax1.tick_params(axis="y", labelcolor=color)
  ax1.set_ylim(-5, 105)
  ax1.grid(True, linestyle="--", alpha=0.6)

  ax2 = ax1.twinx()
  color = "tab:red"
  ax2.set_ylabel("Avg Steps to Sort", color=color, fontsize=12)
  ax2.plot(
      n_values,
      avg_steps,
      color=color,
      marker="s",
      linestyle="--",
      linewidth=2,
      label="Avg Steps",
  )
  ax2.tick_params(axis="y", labelcolor=color)

  plt.title("RL Agent Swap-Sorting Performance Across Curriculum Stages")
  fig.tight_layout()
  plt.show()


# ==========================================
# 4. Main Execution Pipeline
# ==========================================

if __name__ == "__main__":
  MAX_N = 20
  INITIAL_N = 4

  train_env = SwapSortingEnv(max_n=MAX_N, initial_n=INITIAL_N, max_steps=40)
  eval_env = SwapSortingEnv(max_n=MAX_N, initial_n=INITIAL_N, max_steps=MAX_N ** 2)

  curriculum_callback = CurriculumCallback(
      eval_env=eval_env,
      target_success_rate=0.85,
      eval_freq=3000,
      n_eval_episodes=20,
  )

  # Train using MaskablePPO
  model = MaskablePPO(
      "MlpPolicy",
      train_env,
      learning_rate=1e-3,
      gamma=0.99,
      verbose=0,
      seed=42,
  )

  print("Starting Training...")
  model.learn(total_timesteps=1_000_000, callback=curriculum_callback)
  print("Training Complete!\n")

  # Run testing routine across all curriculum levels
  print("Evaluating Agent...")
  eval_stages = list(range(3, MAX_N + 1))
  test_results = evaluate_agent(
      model, eval_env, eval_stages=eval_stages, num_episodes=50
  )

  for n, metrics in test_results.items():
    print(
        f"n={n}: Success Rate = {metrics['success_rate']:.1f}%, Avg Steps ="
        f" {metrics['avg_steps']:.2f}"
    )

  plot_results(test_results)
