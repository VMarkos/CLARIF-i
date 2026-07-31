from cpab import TestConfiguration, plot_coaching_evaluation_results

import gc

if __name__ == '__main__':
    gc.collect()
    bc = 0.4
    setting = TestConfiguration(max_n=20, n_timesteps=10_000_000, succ_rate=0.75, bc=bc, num_envs=12, expert='s')
    setting.run()
    setting.evaluate_agent()
    setting.save_results()
    plot_coaching_evaluation_results({'PPO + Coaching': setting.results}, f'{setting.slug}.png')
    gc.collect()
    bc = 0.8
    setting = TestConfiguration(max_n=20, n_timesteps=10_000_000, succ_rate=0.75, bc=bc, num_envs=12, expert='s')
    setting.run()
    setting.evaluate_agent()
    setting.save_results()
    plot_coaching_evaluation_results({'PPO + Coaching': setting.results}, f'{setting.slug}.png')
    gc.collect()
