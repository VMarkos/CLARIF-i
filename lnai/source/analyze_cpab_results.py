"""Analyse CPAB training logs and evaluation metrics (step 6 mixed budgets).

Default suite (refactored MultiDiscrete CPAB):
  - shared BC=0.0:  es / t=2e6
  - bubble BC>0:    eb / t=2e6
  - selection BC>0: es / t=1e6

Reuses cached ``*_sr.json`` / ``*.json`` eval files when present.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cpab import TestConfiguration, plot_coaching_evaluation_results
from plotters import get_rolling_avg

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
ROLLING = 500
EVAL_EPISODES = 100
MAX_N = 20
SUCC_RATE = 0.75


@dataclass(frozen=True)
class RunSpec:
  expert: str  # 'b' or 's'
  bc: float
  timesteps: int
  label: str


# Step-6 suite from instructions.md
NO_COACH = RunSpec("s", 0.0, 2_000_000, "BC = 0.0 (no coach)")
BUBBLE_RUNS = [
    NO_COACH,
    RunSpec("b", 0.4, 2_000_000, "Bubble (BC=0.4)"),
    RunSpec("b", 0.8, 2_000_000, "Bubble (BC=0.8)"),
]
SELECTION_RUNS = [
    NO_COACH,
    RunSpec("s", 0.4, 1_000_000, "Selection (BC=0.4)"),
    RunSpec("s", 0.8, 1_000_000, "Selection (BC=0.8)"),
]


def slug(expert: str, bc: float, timesteps: int) -> str:
  return f"cpab_res_max_n{MAX_N}_t{timesteps}_sr{SUCC_RATE}_e{expert}_bc{bc}"


def load_monitor_episodes(spec: RunSpec) -> pd.DataFrame:
  pattern = f"{slug(spec.expert, spec.bc, spec.timesteps)}_rank*.monitor.csv"
  frames = []
  for path in sorted(LOG_DIR.glob(pattern)):
    try:
      df = pd.read_csv(path, skiprows=1)
    except Exception:
      continue
    if df.empty or not {"r", "l", "t"}.issubset(df.columns):
      continue
    frames.append(df[["r", "l", "t"]])
  if not frames:
    return pd.DataFrame(columns=["r", "l", "t"])
  return pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)


def plot_training_curves(runs: list[RunSpec], expert_label: str, out_dir: Path) -> None:
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)

  for spec in runs:
    df = load_monitor_episodes(spec)
    if df.empty:
      print(f"[warn] No monitor data for {spec.label} ({slug(spec.expert, spec.bc, spec.timesteps)})")
      continue
    # Legend notes budget when it differs from the group default.
    legend = spec.label
    if spec.bc > 0 and expert_label == "Selection" and spec.timesteps != 2_000_000:
      legend = f"{spec.label} (T={spec.timesteps // 10**6}e6)"
    elif spec.bc > 0 and expert_label == "Bubble" and spec.timesteps != 2_000_000:
      legend = f"{spec.label} (T={spec.timesteps // 10**6}e6)"

    rewards = get_rolling_avg(df["r"].to_numpy(), ROLLING, "valid")
    lengths = get_rolling_avg(df["l"].to_numpy(), ROLLING, "valid")
    axes[0].plot(range(len(rewards)), rewards, linewidth=1.8, label=legend)
    axes[1].plot(range(len(lengths)), lengths, linewidth=1.8, label=legend)
    print(f"[{expert_label}] {spec.label}: episodes={len(df)} T={spec.timesteps}")

  axes[0].set_title(f"Rolling Mean Episode Reward ({expert_label})")
  axes[0].set_xlabel(f"Episode (rolling window = {ROLLING})")
  axes[0].set_ylabel("Mean reward")
  axes[0].grid(True, linestyle="--", alpha=0.5)
  axes[0].legend()

  axes[1].set_title(f"Rolling Mean Episode Length ({expert_label})")
  axes[1].set_xlabel(f"Episode (rolling window = {ROLLING})")
  axes[1].set_ylabel("Mean length")
  axes[1].grid(True, linestyle="--", alpha=0.5)
  axes[1].legend()

  fig.tight_layout()
  out = out_dir / f"training_curves_{expert_label.lower()}.png"
  fig.savefig(out, bbox_inches="tight")
  plt.close(fig)
  print(f"Wrote {out}")


def _normalise_results(results: dict) -> dict:
  return {
      metric: {int(k): float(v) for k, v in series.items()}
      for metric, series in results.items()
  }


def load_or_evaluate(spec: RunSpec, out_dir: Path) -> dict | None:
  name = slug(spec.expert, spec.bc, spec.timesteps)
  candidates = [
      ROOT / f"{name}_sr.json",
      ROOT / f"{name}.json",
      out_dir / f"{name}_eval.json",
      out_dir / f"{name}_sr.json",
  ]
  for json_path in candidates:
    if json_path.exists():
      with open(json_path) as f:
        payload = json.load(f)
      print(f"Loaded eval JSON: {json_path.name}")
      return _normalise_results(payload.get("results", payload))

  zip_path = ROOT / f"{name}.zip"
  alt = ROOT / f"{name}_sr.zip"
  if not zip_path.exists() and alt.exists():
    zip_path = alt
  if not zip_path.exists():
    print(f"[warn] Missing model/JSON for {name}")
    return None

  print(f"Evaluating model {zip_path.name} ({EVAL_EPISODES} eps/n)...")
  setting = TestConfiguration(
      max_n=MAX_N,
      n_timesteps=spec.timesteps,
      succ_rate=SUCC_RATE,
      bc=spec.bc,
      num_envs=1,
      expert=spec.expert,
  )
  setting.load(str(zip_path))
  setting.evaluate_agent(num_episodes=EVAL_EPISODES, criterion="sr")
  out_json = out_dir / f"{name}_eval.json"
  setting.save_results(str(out_json))
  return setting.results


def plot_eval_grouped(expert_label: str, eval_data: dict, out_dir: Path) -> None:
  if not eval_data:
    print(f"[warn] No eval data to plot for {expert_label}")
    return
  out = out_dir / f"eval_efficacy_conformity_{expert_label.lower()}.png"
  plot_coaching_evaluation_results(eval_data, save_path=str(out))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=ROOT / "results" / "cpab" / "t2000000",
  )
  args = parser.parse_args()
  out_dir = args.out_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  plot_training_curves(BUBBLE_RUNS, "Bubble", out_dir)
  plot_training_curves(SELECTION_RUNS, "Selection", out_dir)

  consolidated: dict = {}
  no_coach = load_or_evaluate(NO_COACH, out_dir)
  if no_coach is not None:
    consolidated["NoCoach_bc0.0"] = no_coach

  for expert_label, runs in (("Bubble", BUBBLE_RUNS), ("Selection", SELECTION_RUNS)):
    grouped = {}
    for spec in runs:
      results = load_or_evaluate(spec, out_dir)
      if results is None:
        continue
      grouped[spec.label] = results
      if spec.bc == 0.0:
        continue
      consolidated[f"{expert_label}_bc{spec.bc}"] = results
    plot_eval_grouped(expert_label, grouped, out_dir)

  out_json = out_dir / "cpab_eval_consolidated.json"
  serialisable = {
      name: {
          metric: {str(k): v for k, v in series.items()}
          for metric, series in metrics.items()
      }
      for name, metrics in consolidated.items()
  }
  with open(out_json, "w") as f:
    json.dump(serialisable, f, indent=2)
  print(f"Wrote {out_json}")


if __name__ == "__main__":
  main()
