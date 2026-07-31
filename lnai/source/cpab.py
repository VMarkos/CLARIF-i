# cpab.py

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
# Disable GStreamer plugin loading in OpenCV
os.environ["FLAGS_gstreamer_verbose"] = "0"

import json
import numpy as np
from typing import Callable
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

import matplotlib.pyplot as plt
import seaborn as sns

from lnai.rl.CurriculumSortEnv import CurriculumSortEnv
from lnai.rl.CurriculumPPO_AnnealedBC import CurriculumPPO_AnnealedBC
from lnai.rl.MixedCurriculumCallback import MixedCurriculumCallback
from lnai.rl.Experts import BubbleSortExpert, SelectionSortExpert

from plotters import plot_rolling_reward_plot


def _make_env(setting, rank) -> Callable:
    slug = f'logs/{setting.slug}_rank{rank}'
    def _init():
        env = CurriculumSortEnv(max_n=setting.max_n)
        env = Monitor(env, filename=slug)
        return env
    return _init


class TestConfiguration:
    def __init__(self, max_n: int, n_timesteps: int, succ_rate: float = 0.90, num_envs: int = 8, expert: str='b', bc: float=0.4) -> None:
        self.max_n = max_n
        self.n_timesteps = n_timesteps
        self.succ_rate = succ_rate
        self.num_envs = num_envs
        self.expert_class = BubbleSortExpert if expert == 'b' else SelectionSortExpert
        self.window_scaling = None # Redundant
        self.bc = bc
        self.results = dict()
        self.slug = f'cpab_res_max_n{self.max_n}_t{self.n_timesteps}_sr{self.succ_rate}_e{expert}_bc{bc}'




    def run(self) -> None:
        self.env = SubprocVecEnv([_make_env(self, i) for i in range(self.num_envs)], start_method='spawn')
        #self.env = DummyVecEnv([self._make_env(i) for i in range(self.num_envs)])
        dummy = CurriculumSortEnv(max_n=self.max_n)

        expert = self.expert_class(dummy.pair_to_action)
        curriculum = MixedCurriculumCallback(target_success_rate=self.succ_rate, base_window_size=80, window_update=self.window_scaling)

        self.model = CurriculumPPO_AnnealedBC(
            "MlpPolicy",
            self.env,
            expert=expert,
            initial_bc_coef=self.bc,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=5,
            ent_coef=0.01,
            verbose=1,
        )

        self.model.learn(total_timesteps=self.n_timesteps, callback=curriculum)
        self.model.save(f'{self.slug}.zip')
        self.env.close()


    def _get_bubble_sort_action(self, arr: np.ndarray) -> tuple[int, int] | None:
        """
        Reference coach policy: returns the exact (i, j) swap index pair
        prescribed by standard Bubble Sort for state `arr`.
        Returns None if array is already sorted.
        """
        n = len(arr)
        for i in range(n - 1):
            if arr[i] > arr[i + 1]:
                return (i, i + 1)
        return None  # Already sorted

    def evaluate_agent(
        self,
        num_episodes: int = 5_000,
        max_steps_factor: float = 2.0
    ) -> None:
        """
        Evaluates agent efficacy and conformity across multiple problem sizes (N).

        Returns:
            results = {
                'efficacy': {N: success_rate},
                'conformity': {N: average_conformity},
                'mean_steps': {N: average_steps_taken}
            }
        """
        array_sizes = list(range(2, self.max_n + 1))
        results = {'efficacy': {}, 'conformity': {}, 'mean_steps': {}}

        for n in array_sizes:
            successful_episodes = 0
            total_matched_actions = 0
            total_eval_steps = 0
            episode_steps_list = []

            # Instantiate environment for current array size N
            # env = env_class(n=n, max_steps=int(max_steps_factor * (n ** 2)))
            env = CurriculumSortEnv(max_n=max(array_sizes))
            env.set_max_stage(n)
            expert = self.expert_class()

            for ep in range(num_episodes):
                obs, info = env.reset()
                done = False
                ep_steps = 0

                while not done:
                    # 1. Query agent for action
                    action, _ = self.model.predict(obs, deterministic=True)

                    # 2. Get current state array representation from env
                    current_array = env.get_current_array()

                    # 3. Query coach/expert for ground truth decision at this state
                    coach_action = expert.get_action(current_array, len(current_array))

                    # Convert model action back to swap indices (i, j) if encoded
                    agent_swap = env.action_to_swap(action)

                    # 4. Compute step-level conformity match
                    if coach_action is not None and agent_swap == coach_action:
                        total_matched_actions += 1
                    elif coach_action is None and env.is_sorted(current_array):
                        # Agent state is already sorted
                        total_matched_actions += 1

                    total_eval_steps += 1
                    ep_steps += 1

                    # 5. Environment step
                    obs, reward, done, truncated, info = env.step(action)
                    if done or truncated:
                        if info.get('is_success', False) or env.is_sorted(env.get_current_array()):
                            successful_episodes += 1
                        break

                episode_steps_list.append(ep_steps)

            # Aggregate metrics for dimension N
            results['efficacy'][n] = successful_episodes / num_episodes
            results['conformity'][n] = (
                total_matched_actions / total_eval_steps if total_eval_steps > 0 else 0.0
            )
            results['mean_steps'][n] = np.mean(episode_steps_list)

            print(f"[N={n:02d}] Efficacy: {results['efficacy'][n]*100:.1f}% | "
                  f"Conformity: {results['conformity'][n]*100:.1f}% | "
                  f"Avg Steps: {results['mean_steps'][n]:.1f}")

        self.results = results

    
    def save_results(self) -> None:
        slug = f'{self.slug}.json'
        res = {
            'results': self.results,
        }
        with open(slug, 'w') as file:
            json.dump(res, file)



    def load(self, path: str) -> None:
        self.model = CurriculumPPO_AnnealedBC.load(path)

def plot_coaching_evaluation_results(
    eval_data: dict[str, dict[str, dict[int, float]]],
    save_path: str = "coaching_evaluation_metrics.png"
):
    """
    Plots Efficacy and Conformity across array sizes N for one or multiple algorithms/models.

    `eval_data` format:
    {
        'PPO + Coaching': {'efficacy': {3: 1.0, 5: 0.95, ...}, 'conformity': {3: 0.98, 5: 0.82, ...}},
        'Pure PPO (No Coach)': {'efficacy': {3: 1.0, 5: 0.40, ...}, 'conformity': {3: 0.30, 5: 0.12, ...}}
    }
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Color palette & marker styling
    colors = sns.color_palette("Set2", len(eval_data))
    markers = ['o', 's', '^', 'D', 'v']

    # Subplot 1: Efficacy vs N
    ax1 = axes[0]
    sizes = []
    for idx, (label, metrics) in enumerate(eval_data.items()):
        sizes = sorted(list(metrics['efficacy'].keys()))
        efficacies = [metrics['efficacy'][n] * 100 for n in sizes]
        ax1.plot(
            sizes, efficacies,
            label=label,
            marker=markers[idx % len(markers)],
            color=colors[idx],
            linewidth=2,
            markersize=7
        )

    ax1.set_title("Sorting Efficacy ($E$)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Array Size ($N$)", fontsize=11)
    ax1.set_ylabel("Success Rate (%)", fontsize=11)
    ax1.set_ylim(-5, 105)
    ax1.set_xticks(sizes)
    ax1.legend(frameon=True, facecolor='white', edgecolor='none')

    # Subplot 2: Conformity vs N
    ax2 = axes[1]
    for idx, (label, metrics) in enumerate(eval_data.items()):
        sizes = sorted(list(metrics['conformity'].keys()))
        conformities = [metrics['conformity'][n] * 100 for n in sizes]
        ax2.plot(
            sizes, conformities,
            label=label,
            marker=markers[idx % len(markers)],
            color=colors[idx],
            linewidth=2,
            markersize=7
        )

    ax2.set_title("Coach Conformity ($C$)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Array Size ($N$)", fontsize=11)
    ax2.set_ylabel("Conformity Rate (%)", fontsize=11)
    ax2.set_ylim(-5, 105)
    ax2.set_xticks(sizes)
    ax2.legend(frameon=True, facecolor='white', edgecolor='none')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()
    print(f"Evaluation plot successfully saved to {save_path}")
