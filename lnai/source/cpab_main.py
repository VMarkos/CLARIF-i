from cpab import TestConfiguration, plot_coaching_evaluation_results


if __name__ == '__main__':

    for bc in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        print(f'>>> Running bc={bc}')
        setting = TestConfiguration(max_n=10, n_timesteps=100_000, bc=bc)
        setting.run()
        # setting.load('cpab_N12_T5000000_SR0.9.zip')
        setting.evaluate_agent()
        setting.save_results()
        plot_coaching_evaluation_results({'PPO + Coaching': setting.results}, f'{setting.slug}.png')
