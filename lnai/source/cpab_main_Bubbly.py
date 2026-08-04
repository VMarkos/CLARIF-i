from cpab_Bubbly import TestConfiguration_Bubble, plot_coaching_evaluation_results_Bubble

import gc

if __name__ == '__main__':
    gc.collect()
    criterion = 'i'
    for bc in (0.0, 0.4, 0.8):
        setting = TestConfiguration_Bubble(
            max_n=20,
            n_timesteps=1_000_000,
            succ_rate=0.75,
            bc=bc,
            num_envs=4,
            expert='b',
        )
        # setting.load(f'{setting.slug}.zip')
        setting.run()
        setting.evaluate_agent(100, criterion=criterion)
        setting.save_results(f'{setting.slug}_{criterion}.json')
        plot_coaching_evaluation_results_Bubble(
            {'PPO + Coaching (Bubble)': setting.results},
            f'{setting.slug}_{criterion}.png',
        )
        gc.collect()
