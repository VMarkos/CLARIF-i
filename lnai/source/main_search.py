# main_search.py
#
# temp script

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

# Import the CurriculumPathSearchEnv implemented earlier
# from env import CurriculumPathSearchEnv
from lnai.rl.PathSearchEnv import CurriculumPathSearchEnv
from lnai.rl.CurriculumCallback import CurriculumCallback

if __name__ == "__main__":
    # 1. Initialize environment with max capacity N=6, starting at N=3
    env = CurriculumPathSearchEnv(max_n=6, current_n=3, max_path_len=20)

    # 2. Instantiate MaskablePPO
    model = MaskablePPO(
        MaskableMultiInputActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        verbose=0  # Suppress default log noise to highlight curriculum output
    )

    # 3. Create Curriculum Callback
    curriculum_callback = CurriculumCallback(
        eval_freq=1500,           # Evaluate every 1500 steps
        n_eval_episodes=20,       # Average over 20 runs
        success_threshold=0.80,   # Level up at 80% success rate
        verbose=1
    )

    # 4. Train model across dynamic levels
    print("Starting Curriculum Training (Starting at n=3)...")
    model.learn(total_timesteps=30000, callback=curriculum_callback)
