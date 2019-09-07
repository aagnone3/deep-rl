#!/usr/bin/env python
# coding: utf-8

# # Navigation
# In this notebook, you will learn how to use the Unity ML-Agents environment for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
# ### 1. Start the Environment
import os
import sys
from os import path
from collections import deque
from datetime import datetime

import torch
import numpy as np
from unityagents import UnityEnvironment
# from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.getcwd())
CKPT_DIR = path.join(path.dirname(__file__), 'checkpoints')

from dqn_agent import Agent

env = UnityEnvironment(file_name="p1_navigation/Banana_Linux_NoVis/Banana.x86_64")

# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents.
# Here we check for the first brain available, and set it as the default brain we will be controlling from Python.
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Examine the State and Action Spaces
# The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
# - `0` - walk forward
# - `1` - walk backward
# - `2` - turn left
# - `3` - turn right
#
# The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects
# around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward
# of `-1` is provided for collecting a blue banana.

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
print('Number of agents:', len(env_info.agents))
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

agent = Agent(state_size=state_size, action_size=action_size, seed=0, mode='Double_DQN', use_prioritized_memory=True)


def dqn(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        while 1:
            action = agent.get_action(state, eps)          # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done) # take step with agent (including learning)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon for next episode

        # Printing & Monitoring
        cur_mean = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, cur_mean), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tepsilon: {:.5f}'.format(i_episode, cur_mean,eps))

        # Environment is solved if average score of last 100 episodes >= 13
        if cur_mean >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, cur_mean))
            file_basename = path.join(CKPT_DIR, 'solved_{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
            torch.save(agent.qnetwork_local.state_dict(), file_basename + '.pt')
            np.savez_compressed(file_basename + '.npz', scores=scores)
            break
    return scores

n_episodes_max = 1000
eps_end = .005
eps_decay = .995

scores = dqn(n_episodes=n_episodes_max, eps_end=eps_end, eps_decay=eps_decay)
n_episodes = len(scores)
print('# episodes: {}'.format(n_episodes))

env.close()
