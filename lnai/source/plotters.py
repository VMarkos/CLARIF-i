# plotters.py
#
# Plotting utilities


import numpy as np
from matplotlib import pyplot as plt



def get_rolling_avg(arr, window, convolution_mode) -> np.ndarray:
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode,
    ) / window


def plot_rolling_reward_plot(
        env,
        rolling_length: int=500,
    ) -> None:
    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
    
    # Rewards
    ax[0].set_title('Episode Rewards')
    reward_rolling_avg = get_rolling_avg(
        env.return_queue,
        rolling_length,
        'valid',
    )
    ax[0].plot(range(len(reward_rolling_avg)), reward_rolling_avg)
    ax[0].set_ylabel('Avg reward')
    ax[0].set_xlabel('Episode')

    # Episode length
    ax[1].set_title('Episode Length')
    episode_rolling_avg = get_rolling_avg(
        env.length_queue,
        rolling_length,
        'valid',
    )
    ax[1].plot(range(len(episode_rolling_avg)), episode_rolling_avg)
    ax[1].set_ylabel('Avg episode length')
    ax[1].set_xlabel('Episode')
    
    # Training error
    # ax[2].set_title('Training Error')
    # error_rolling_avg = get_rolling_avg(
    #     agent.training_error,
    #     rolling_length,
    #     'same',
    # )
    # ax[2].plot(range(len(error_rolling_avg)), error_rolling_avg)
    # ax[2].set_ylabel('Avg training error')
    # ax[2].set_xlabel('Step')

    plt.tight_layout()
    plt.show()
    
