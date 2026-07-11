# main.py

from lnai.rl.SortingEnv import SortingEnv


def main():
    env = SortingEnv(8)
    env.reset() # Is this actually needed?
    env.show()


if __name__ == '__main__':
    main()
