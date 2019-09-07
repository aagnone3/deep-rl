#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        dest='in_file',
        help='Path to .npz file of rewards array, expected under the key "scores".'
    )
    parser.add_argument(
        dest='out_file',
        help='Path where created plot should be persisted.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.in_file)['scores']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data, label='raw')
    ax.plot(pd.Series(data).ewm(span=10).mean(), label='EMA-10')
    plt.xlabel('Episode')
    plt.ylabel('Moving Avg Reward')
    plt.grid(True)
    plt.legend()
    fig.savefig(args.out_file)


if __name__ == '__main__':
    main()
