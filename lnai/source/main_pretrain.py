# main_pretrain.py
#
# temp script


from gymnasium.wrappers import RecordEpisodeStatistics
from lnai.rl.Pretrainer import pretrain_dagger
from lnai.rl.SortingEnv import SortingEnv
from lnai.rl.SortingCallback import SortingCallback
from parsers import get_ql_parser
from plotters import plot_rolling_reward_plot
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from utils import bubble_sort, quick_sort

def main() -> None:
    parser = get_ql_parser()
    args = parser.parse_args()
    env = SortingEnv(args.n)
    env = RecordEpisodeStatistics(env, args.s)
    callback = SortingCallback(end_n=args.n, verbose=1, reward_thresh=10.5, start_n=3)
    model = MaskablePPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=args.s, callback=callback)
    model.save(f'pretrained_{args.n}_{args.s}.zip')
    obs, info = env.reset()
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
