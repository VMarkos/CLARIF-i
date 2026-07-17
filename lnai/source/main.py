# main.py

from gymnasium.wrappers import RecordEpisodeStatistics
from lnai.rl.SortingEnv import SortingEnv
from lnai.rl.SortingCallback import SortingCallback
from plotters import plot_rolling_reward_plot
from parsers import get_ql_parser
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
# from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.env import MaskableEnvWrapper
# from stable_baselines3.common.env_util import make_vec_env


def main():
    parser = get_ql_parser()
    args = parser.parse_args()
    n = args.n
    n_timesteps = args.s
    load = args.l
    env = MaskableEnvWrapper(SortingEnv(n))
    # env = gym.make('CartPole-v1', render_mode='human')
    env = RecordEpisodeStatistics(env, n_timesteps)
    callback = SortingCallback(end_n=n)
    # Create and train PPO model
    if load:
        model = PPO.load(load, env=env)
    else:
        model = MaskablePPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=n_timesteps)
        model.save(f'ppo_{n}_{n_timesteps}.zip')
        # Plot learning outputs
        plot_rolling_reward_plot(env, rolling_length=500, fname=f'ppo_{n}_{n_timesteps}.pdf')


    # Test model
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
