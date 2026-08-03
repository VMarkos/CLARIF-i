import itertools
import math
import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# ==========================================
# 1. Levenshtein Distance Helper
# ==========================================


def levenshtein_distance(seq1, seq2):
  """Computes standard Levenshtein (edit) distance between two sequences of actions."""
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
        dp[i][j] = 1 + min(
            dp[i - 1][j],  # Deletion
            dp[i][j - 1],  # Insertion
            dp[i - 1][j - 1],  # Substitution
        )

  return dp[m][n]


# ==========================================
# 2. Custom Gymnasium Environment with Coaching
# ==========================================


class CoachedSwapSortingEnv(gym.Env):

  def __init__(self, max_n=10, initial_n=3, max_steps=40):
    super().__init__()
    self.max_n = max_n
    self.current_n = initial_n
    self.max_steps = max_steps
    self.steps_taken = 0

    self.observation_space = spaces.Box(
        low=1, high=self.max_n, shape=(self.max_n,), dtype=np.int32
    )

    # Precompute mapping for C(max_n, 2) unique swap pairs (i, j) where i < j
    self.swap_pairs = list(itertools.combinations(range(self.max_n), 2))
    self.pair_to_action = {pair: idx for idx, pair in enumerate(self.swap_pairs)}
    self.num_actions = len(self.swap_pairs)
    self.action_space = spaces.Discrete(self.num_actions)

    self.state = None
    self.agent_trace = []
    self.coach_trace = []
    self.prev_distance = 0
    self.prev_inversions = 0

  def set_n(self, n):
    """Update current curriculum stage n."""
    self.current_n = min(n, self.max_n)
    self.max_steps = max(40, 2 * self.current_n ** 2)

  def action_masks(self):
    """Mask actions that involve indices >= current_n."""
    mask = np.zeros(self.num_actions, dtype=bool)
    for idx, (i, j) in enumerate(self.swap_pairs):
      if i < self.current_n and j < self.current_n:
        mask[idx] = True
    return mask

  def _generate_bubble_sort_trace(self, arr_slice):
    """Generates the reference swap trace using Bubble Sort on the active array slice."""
    arr = list(arr_slice)
    trace = []
    n = len(arr)

    for i in range(n):
      for j in range(0, n - i - 1):
        if arr[j] > arr[j + 1]:
          arr[j], arr[j + 1] = arr[j + 1], arr[j]
          pair = (j, j + 1)
          trace.append(self.pair_to_action[pair])

    return trace

  def _count_inversions(self, arr):
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
    self.agent_trace = []

    # Initialize active slice
    active_slice = np.arange(1, self.current_n + 1, dtype=np.int32)
    while True:
      self.np_random.shuffle(active_slice)
      if self._count_inversions(active_slice) > 0 or self.current_n <= 1:
        break

    self.state = np.arange(1, self.max_n + 1, dtype=np.int32)
    self.state[: self.current_n] = active_slice

    # Generate coach trace
    self.coach_trace = self._generate_bubble_sort_trace(
        self.state[: self.current_n]
    )

    # Initial Levenshtein distance
    self.prev_distance = levenshtein_distance(self.agent_trace, self.coach_trace)
    self.prev_inversions = self._count_inversions(self.state)

    return self.state.copy(), {}

  def step(self, action):
    self.steps_taken += 1
    self.agent_trace.append(action)

    i, j = self.swap_pairs[action]
    self.state[i], self.state[j] = self.state[j], self.state[i]

    # Calculate trace distance to coach
    current_distance = levenshtein_distance(self.agent_trace, self.coach_trace)

    # Reward shaping based on distance reduction
    delta_distance = self.prev_distance - current_distance
    self.prev_distance = current_distance

    reward = float(delta_distance) * 2.0 - 0.1  # Living penalty

    inversions = self._count_inversions(self.state)
    delta_inversions = self.prev_inversions - inversions
    self.prev_inversions = inversions

    reward += float(delta_inversions) * 2.0

    # Check for termination
    terminated = inversions == 0

    if terminated:
      reward += 20.0  # Completion bonus

    truncated = self.steps_taken >= self.max_steps

    info = {
        "coach_trace": self.coach_trace,
        "agent_trace": self.agent_trace,
        "final_distance": current_distance,
    }

    return self.state.copy(), reward, terminated, truncated, info


# ==========================================
# 3. Picklable Top-Level Env Factory
# ==========================================


def make_coached_env(max_n, initial_n, max_steps):
  """Top-level factory function required for SubprocVecEnv process serialization."""

  def _thunk():
    return CoachedSwapSortingEnv(
        max_n=max_n, initial_n=initial_n, max_steps=max_steps
    )

  return _thunk


# ==========================================
# 4. SB3 Curriculum Callback
# ==========================================


class CurriculumCallback(BaseCallback):

  def __init__(
      self,
      eval_env,
      target_success_rate=0.85,
      eval_freq=3000,
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
      self.n_eval_episodes = 2 * self.current_n ** 2
      for _ in range(self.n_eval_episodes):
        obs, _ = self.eval_env.reset()
        done = False
        while not done:
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
            f"[Step {self.num_timesteps}] Stage n={self.current_n} | Success"
            f" Rate: {success_rate:.2f}"
        )

      if (
          success_rate >= self.target_success_rate
          and self.current_n < self.eval_env.max_n
      ):
        self.current_n += 1
        if self.verbose > 0:
          print(f"--> Advancing Curriculum Stage to n = {self.current_n}!")

        # Broadcast stage change across all parallel SubprocVecEnv worker environments
        self.training_env.env_method("set_n", self.current_n)
        self.eval_env.set_n(self.current_n)

    return True


# ==========================================
# 5. Evaluation Routine & Plotting
# ==========================================


def evaluate_coached_agent(model, env, eval_stages, num_episodes=50):
  results = {}

  for n in eval_stages:
    env.set_n(n)
    successes = 0
    fidelities = []
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
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

      if terminated:
        successes += 1
        step_counts.append(steps)

      agent_tr = info["agent_trace"]
      coach_tr = info["coach_trace"]
      dist = levenshtein_distance(agent_tr, coach_tr)
      max_len = max(len(agent_tr), len(coach_tr))

      fidelity = (1.0 - (dist / max_len)) if max_len > 0 else 1.0
      fidelities.append(fidelity)

    results[n] = {
        "success_rate": (successes / num_episodes) * 100,
        "avg_steps": np.mean(step_counts) if step_counts else env.max_steps,
        "mean_fidelity": np.mean(fidelities) * 100,
    }

  return results


def plot_coaching_results(results):
  n_values = list(results.keys())
  success_rates = [results[n]["success_rate"] for n in n_values]
  fidelities = [results[n]["mean_fidelity"] for n in n_values]
  avg_steps = [results[n]["avg_steps"] for n in n_values]

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

  # Plot 1: Success Rate & Fidelity
  ax1.plot(
      n_values,
      success_rates,
      color="tab:blue",
      marker="o",
      linewidth=2,
      label="Sorting Success Rate (%)",
  )
  ax1.plot(
      n_values,
      fidelities,
      color="tab:green",
      marker="^",
      linestyle="--",
      linewidth=2,
      label="Bubble Sort Fidelity (%)",
  )
  ax1.set_xlabel("List Size (n)", fontsize=11)
  ax1.set_ylabel("Percentage (%)", fontsize=11)
  ax1.set_ylim(-5, 105)
  ax1.set_title("Agent Accuracy & Coach Fidelity")
  ax1.grid(True, linestyle="--", alpha=0.6)
  ax1.legend(loc="lower left")

  # Plot 2: Average Steps
  ax2.plot(
      n_values,
      avg_steps,
      color="tab:red",
      marker="s",
      linewidth=2,
      label="Avg Steps Taken",
  )
  ax2.set_xlabel("List Size (n)", fontsize=11)
  ax2.set_ylabel("Steps", fontsize=11)
  ax2.set_title("Average Steps to Solution")
  ax2.grid(True, linestyle="--", alpha=0.6)
  ax2.legend(loc="upper left")

  fig.suptitle(
      "Coached RL Agent (Parallel SubprocVecEnv - 8 Workers)",
      fontsize=14,
      fontweight="bold",
  )
  fig.tight_layout()
  plt.show()


# ==========================================
# 6. Main Execution Pipeline
# ==========================================

if __name__ == "__main__":
  MAX_N = 20
  INITIAL_N = 3
  NUM_ENVS = 4  # 8 Parallel Environments

  # Create vectorized training environment using SubprocVecEnv
  train_env = make_vec_env(
      make_coached_env(max_n=MAX_N, initial_n=INITIAL_N, max_steps=40),
      n_envs=NUM_ENVS,
      vec_env_cls=SubprocVecEnv,
      seed=42,
  )

  # Single evaluation environment
  eval_env = CoachedSwapSortingEnv(
      max_n=MAX_N, initial_n=INITIAL_N, max_steps=40
  )

  curriculum_callback = CurriculumCallback(
      eval_env=eval_env,
      target_success_rate=0.85,
      eval_freq=1000,  # Evaluates every 1000 calls (8000 total timesteps)
      n_eval_episodes=20,
  )

  model = MaskablePPO(
      "MlpPolicy",
      train_env,
      learning_rate=1e-3,
      gamma=0.99,
      n_steps=256,  # Rollout steps per environment
      batch_size=64,
      verbose=0,
      seed=42,
  )

  print(
      f"Starting Coached RL Training with SubprocVecEnv ({NUM_ENVS} workers)..."
  )
  model.learn(total_timesteps=10_000_000, callback=curriculum_callback)
  print("Training Complete!\n")
  model.save('final_lev.zip')

  # Run evaluation routine
  print("Evaluating Coached Agent...")
  eval_stages = list(range(3, MAX_N + 1))
  test_results = evaluate_coached_agent(
      model, eval_env, eval_stages=eval_stages, num_episodes=50
  )

  for n, metrics in test_results.items():
    print(
        f"n={n}: Success Rate = {metrics['success_rate']:.1f}% | "
        f"Bubble Sort Fidelity = {metrics['mean_fidelity']:.1f}% | "
        f"Avg Steps = {metrics['avg_steps']:.2f}"
    )

  plot_coaching_results(test_results)
