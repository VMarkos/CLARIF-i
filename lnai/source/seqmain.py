import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from lnai.rl.SortingCoach import SortingCoach
from lnai.rl.SortingEnv import SortingEnv
from lnai.rl.SeqTransEnv import SeqTransEnv

from utils import bubble_sort


# ---------------------------------------------------------------------------
# 1. Canonical Reference Sorting Algorithm (Coach Oracle)
# ---------------------------------------------------------------------------
'''
def bubble_sort_algorithm(arr: np.ndarray) -> np.ndarray:
    """Standard Bubble Sort algorithm for the SortingCoach."""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
'''


# ---------------------------------------------------------------------------
# 2. Helper Wrapper for Action Masking Compatibility
# ---------------------------------------------------------------------------
def mask_fn(env: SeqTransEnv) -> np.ndarray:
    """Extracts valid action mask from SeqTransEnv for SB3's ActionMasker wrapper."""
    return env.action_masks()


# ---------------------------------------------------------------------------
# 3. Main Execution Routine
# ---------------------------------------------------------------------------
def main():
    load = True
    MAX_N = 5  # Array size for testing
    
    print("=== Step 1: Initializing Sorter Model & Coach ===")
    # Initialize base sorting environment and Sorter PPO model
    sort_env = SortingEnv(MAX_N)
    '''sorter_model = MaskablePPO(
        "MlpPolicy",
        sort_env,
        verbose=0
    )
    # Quick mock training for sorter (or load your pretrained model here)
    sorter_model.learn(total_timesteps=1000)'''
    sorter_model = MaskablePPO.load('b_ppo_5_30000_0.05.zip')

    # Initialize Coach with Bubble Sort algorithm
    coach = SortingCoach(algorithm=bubble_sort)

    print("\n=== Step 2: Instantiating Sequence Transformation Environment ===")
    raw_env = SeqTransEnv(max_n=MAX_N, sorter=sorter_model, coach=coach)
    
    # Wrap environment with SB3's ActionMasker
    env = ActionMasker(raw_env, mask_fn)

    if not load:
        print("\n=== Step 3: Training Transcoder Agent (MaskablePPO) ===")
        transcoder_model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy,
            env,
            learning_rate=1e-3,
            gamma=0.99,
            verbose=1
        )

        # Short training run as a sanity check
        transcoder_model.learn(total_timesteps=30_000)
        print("Training phase complete!")

        transcoder_model.save('last_transcoder.zip')
    else:
        transcoder_model = MaskablePPO.load('last_transcoder.zip')

    print("\n=== Step 4: Running Test Evaluation Episode ===")
    obs, _ = env.reset()
    done = False
    step_count = 0

    print("\n--- Episode Initial State ---")
    print(f"Source Trace (Model A) Len: {len(raw_env._current_trace)}")
    print(f"    Trace                 : {raw_env._current_trace}")
    print(f"Target Trace (Coach)   Len: {len(raw_env._target_trace)}")
    print(f"    Trace                 : {raw_env._target_trace}")
    print(f"Initial Distance          : {raw_env._prev_distance}")

    terminated, truncated = False, False

    while not done and step_count < 20:
        step_count += 1
        
        # Predict masked action
        action_masks = env.action_masks()
        action, _ = transcoder_model.predict(obs, action_masks=action_masks, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Print step breakdown
        print(f"\nStep {step_count}: Action Taken = {action} | Reward = {reward:.3f}")
        raw_env.render()
        print(f"Current Distance to Target: {raw_env._prev_distance}")

    print("\n=== Final Test Summary ===")
    if terminated:
        print("Success! Model B successfully transformed the trajectory into the Coach's target trace!")
    else:
        print(f"Episode finished via truncation. Final trace distance: {raw_env._prev_distance}")


if __name__ == "__main__":
    main()
