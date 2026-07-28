# main_search.py
#
# temp script

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

# Import the CurriculumPathSearchEnv implemented earlier
from lnai.rl.PathSearchEnv import CurriculumPathSearchEnv
from lnai.rl.CurriculumCallback import CurriculumCallback

# Import arg parser
from parsers import get_ql_parser

if __name__ == "__main__":
    parser = get_ql_parser()
    args = parser.parse_args()
    # 1. Initialize environment with max capacity N=6, starting at N=3
    env = CurriculumPathSearchEnv(max_n=args.n, current_n=3, max_path_len=20)

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
    model.learn(total_timesteps=args.s, callback=curriculum_callback)
    model.save(f'cur_model_n{args.n}_s{args.s}.zip')

    # Test model
    obs, info = env.reset()
    env.render()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            print(f'Done in step {i}!')
            break
    env.close()
