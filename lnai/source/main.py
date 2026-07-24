# main.py

from gymnasium.wrappers import RecordEpisodeStatistics
from lnai.rl.SortingEnv import SortingEnv
from lnai.rl.AlgorithmicSortingEnv import AlgorithmicSortingEnv
from lnai.rl.SortingCallback import SortingCallback
from plotters import plot_rolling_reward_plot
from parsers import get_ql_parser
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from utils import bubble_sort


def main():
    parser = get_ql_parser()
    args = parser.parse_args()
    n = args.n
    n_timesteps = args.s
    load = args.l
    # env = gym.make('CartPole-v1', render_mode='human')
    env = AlgorithmicSortingEnv(bubble_sort, 0.8, n, start_size=2)
    env = RecordEpisodeStatistics(env, n_timesteps)
    callback = SortingCallback(end_n=n, verbose=1, reward_thresh=0.98)
    # Create and train PPO model
    if load:
        model = MaskablePPO.load(load, env=env)
    else:
        model = MaskablePPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=n_timesteps, callback=callback)
        model.save(f'algo_ppo_{n}_{n_timesteps}.zip')
        # Plot learning outputs
        plot_rolling_reward_plot(env, rolling_length=500, fname=f'algo_ppo_{n}_{n_timesteps}.pdf')


    # Test model
    env = AlgorithmicSortingEnv(bubble_sort, 0.0, n, start_size=n)
    obs, info = env.reset()
    # obs = State(dict(zip(range(n),range(n-1,-1,-1)))).as_ndarray()
    env.render()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            print(f'Done in step {i+1}!')
            break
    env.close()


if __name__ == '__main__':
    main()
