# main.py

from lnai.rl.SortingEnv import SortingEnv


def main():
    env = SortingEnv(5)
    env.reset() # Is this actually needed?
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        env_str = env.render()
        print(env_str)
    env.close()


if __name__ == '__main__':
    main()
