from cpab import TestConfiguration, plot_coaching_evaluation_results

from itertools import product

import gc

if __name__ == '__main__':
    gc.collect()
    criterion = 'sr'
    for bc, e in product((1.2, 1.6), ('b', 's')):
        T = 2_000_000 if e == 'b' else 1_000_000
        setting = TestConfiguration(max_n=20, n_timesteps=T, succ_rate=0.75, bc=bc, num_envs=8, expert=e)
        setting.run()
        # setting.load(f'{setting.slug}.zip')
        setting.evaluate_agent(100, criterion=criterion)
        setting.save_results(f'{setting.slug}_{criterion}.json')
        plot_coaching_evaluation_results({'PPO + Coaching': setting.results}, f'{setting.slug}_{criterion}.png')
        gc.collect()
