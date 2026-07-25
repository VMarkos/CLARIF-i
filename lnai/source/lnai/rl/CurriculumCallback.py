# rl/CurriculumCallback.py
#
# Naive curriculum sb3 compatible callback

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    """
    Automated performance-based Curriculum Callback.
    Evaluates success rate on the current 'n' and advances the curriculum level.
    """
    def __init__(
        self,
        eval_freq: int = 2000,
        n_eval_episodes: int = 20,
        success_threshold: float = 0.85,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.success_threshold = success_threshold

    def _on_step(self) -> bool:
        # Run evaluation every `eval_freq` steps
        if self.n_calls % self.eval_freq != 0:
            return True

        # Extract unwrapped environment to access custom set_curriculum_level method
        training_env = self.training_env.envs[0].unwrapped
        current_n = training_env.current_n
        max_n = training_env.max_n

        # If already at max level, skip curriculum logic
        if current_n >= max_n:
            return True

        # Run Evaluation Episodes
        successes = 0
        for _ in range(self.n_eval_episodes):
            obs, _ = training_env.reset()
            done = False
            
            while not done:
                # Retrieve current action mask for MaskablePPO prediction
                action_masks = training_env.action_masks()
                action, _ = self.model.predict(
                    obs, 
                    action_masks=action_masks, 
                    deterministic=True
                )
                obs, reward, terminated, truncated, info = training_env.step(action)
                done = terminated or truncated
                
                # Check if state reached target (terminated without timing out)
                if terminated and not info.get("invalid_action", False):
                    successes += 1

        success_rate = successes / self.n_eval_episodes

        if self.verbose > 0:
            print(f"\n[Curriculum Eval @ Step {self.num_timesteps}] "
                  f"Level n={current_n} | Success Rate: {success_rate:.2%}")

        # Check threshold trigger
        if success_rate >= self.success_threshold:
            new_n = current_n + 1
            if new_n >= 6:
                self.model.ent_coef = 0.05
            self.training_env.env_method('set_curriculum_level', new_n)
            
            # Log curriculum metric to tensorboard/logger
            self.logger.record("curriculum/level", new_n)
            
            if self.verbose > 0:
                print(f"🚀 CURRICULUM UPGRADE: Success threshold reached! "
                      f"Advancing to n = {new_n}\n")

        return True
