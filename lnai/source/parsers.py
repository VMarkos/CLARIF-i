# parsers.py
#
# Argument parsers using argparse

from argparse import ArgumentParser


def get_ql_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog='Sorting Q-Learning',
        description='Q-Learning based sorting agents',
        epilog='This was the help page for the Sorting Q-Learning agent script.'
    )
    parser.add_argument(
        '-n',
        type=int,
        default=5,
        help='Number of integers each state should contain, starting from 0. Defaults to 5',
    )
    parser.add_argument(
        '-s',
        type=int,
        default=100_000,
        help='Number of learning timesteps. Defaults to 100_000.',
    )
    parser.add_argument(
        '-l',
        type=str,
        default='',
        help='Path for pre-trained model to load, Defaults to \'\'.'
    )
    parser.add_argument(
        '-p',
        type=float,
        default=0.05,
        help='Absolute penalty that should be applied in non-conformant actions. Defaults to 0.05.'
    )
    parser.add_argument(
        '-a',
        type=str,
        default='b',
        help='Algorithm to use. Defaults to "b".'
    )
    return parser
