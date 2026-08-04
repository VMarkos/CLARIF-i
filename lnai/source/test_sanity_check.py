"""Sanity-check evaluation for ppo_20_1000000.zip on SortingEnv."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO

from lnai.rl.SortingEnv import SortingEnv

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ppo_20_1000000.zip"
OUT_DIR = ROOT / "results" / "sanity_check"
MAX_N = 20
START_N = 3


def episodes_for_n(n: int) -> int:
  return max(100, 2 * n * n)


def evaluate(model: MaskablePPO, env: SortingEnv, stages: list[int]) -> dict:
  results = {}

  for n in stages:
    env.set_n(n)
    env._max_steps = max(n * n, 100)
    num_episodes = episodes_for_n(n)
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

    success_rate = (successes / num_episodes) * 100.0
    avg_steps = (
        float(np.mean(step_counts)) if step_counts else float(env._max_steps)
    )
    results[n] = {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "num_episodes": num_episodes,
        "successes": successes,
    }
    print(
        f"n={n}: episodes={num_episodes} | "
        f"Success Rate = {success_rate:.1f}% | "
        f"Avg Steps = {avg_steps:.2f}"
    )

  return results


def save_plot(results: dict, path: Path) -> None:
  n_values = list(results.keys())
  success_rates = [results[n]["success_rate"] for n in n_values]
  avg_steps = [results[n]["avg_steps"] for n in n_values]

  fig, ax1 = plt.subplots(figsize=(8, 4.5))
  color = "tab:blue"
  ax1.set_xlabel("List Size (n)", fontsize=12)
  ax1.set_ylabel("Success Rate (%)", color=color, fontsize=12)
  ax1.plot(n_values, success_rates, color=color, marker="o", linewidth=2)
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
  )
  ax2.tick_params(axis="y", labelcolor=color)

  plt.title("Sanity Check: MaskablePPO on SortingEnv (ppo_20_1000000)")
  fig.tight_layout()
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  OUT_DIR.mkdir(parents=True, exist_ok=True)

  env = SortingEnv(max_n=MAX_N, start_size=START_N)
  print(f"Loading {MODEL_PATH} into SortingEnv(max_n={MAX_N}) ...")
  model = MaskablePPO.load(str(MODEL_PATH), env=env)

  stages = list(range(START_N, MAX_N + 1))
  print("Evaluating...")
  results = evaluate(model, env, stages)

  json_path = OUT_DIR / "ppo_20_1000000_eval.json"
  with open(json_path, "w") as f:
    json.dump({"results": {str(k): v for k, v in results.items()}}, f, indent=2)
  print(f"Wrote {json_path}")

  plot_path = OUT_DIR / "ppo_20_1000000_performance.png"
  save_plot(results, plot_path)
  print(f"Wrote {plot_path}")


if __name__ == "__main__":
  main()
