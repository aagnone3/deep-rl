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
    scores = np.load(args.in_file, allow_pickle=True)['arr_0'].item()
    mavgs = scores['moving_avgs']
    means = scores['mean_scores']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(means, label='raw')
    ax.plot(mavgs, label='Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Agent Rewards')
    plt.grid(True)
    plt.legend()

    fig.savefig(args.out_file)


if __name__ == '__main__':
    main()
