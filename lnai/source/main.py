# main.py

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from lnai.api.State import State
from lnai.rl.SortingEnv import SortingEnv
from lnai.rl.SortingAgent import SortingAgent
from plotters import plot_rolling_reward_plot
from parsers import get_ql_parser
from stable_baselines3 import PPO


def main():
    parser = get_ql_parser()
    args = parser.parse_args()
    n = args.n
    n_timesteps = args.s
    load = args.l
    env = SortingEnv(n)
    # env = gym.make('CartPole-v1', render_mode='human')
    env = RecordEpisodeStatistics(env, n_timesteps)
    '''q_learn_hyperparams = {
        'learning_rate':  0.01,
        'initial_epsilon': 1.0,
        'epsilon_decay': 1.0 / (n_episodes / 2),
        'final_epsilon': 0.1,
        'discount_factor': 0.95,
        'n_actions': env.action_space.n,
    }'''
    # Create and train PPO model
    if load:
        model = PPO.load(load, env=env)
    else:
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=n_timesteps)
        model.save('test_ppo.zip')
        # Plot learning outputs
        plot_rolling_reward_plot(env, rolling_length=500)


    # Test model
    vec_env = model.get_env()
    obs = vec_env.reset()
    obs = State(dict(zip(range(n),range(n-1,-1,-1)))).as_ndarray()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        if done:
            print(f'Done in step {i+1}!')
            break
    # agent = SortingAgent(n, n_episodes, env, q_learn_hyperparams)
    # agent.learn()
    env.close()


if __name__ == '__main__':
    main()
