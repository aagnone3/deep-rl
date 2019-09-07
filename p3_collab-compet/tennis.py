#!/usr/bin/env python
# coding: utf-8

# # Collaboration and Competition
import os
import sys

from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
from unityagents import UnityEnvironment

sys.path.insert(0, os.path.dirname(__file__))
from constants import *
from ddpg_agent import Agent
from replay_buffer import ReplayBuffer

base_dir = 'p3_collab-compet'
env = UnityEnvironment(file_name="{}/Tennis_Linux_NoVis/Tennis.x86".format(base_dir))
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
n_agents = len(env_info.agents)
print('Number of agents:', n_agents)

action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
env_info = env.reset(train_mode=True)[brain_name]


def ddpg(n_episodes=1000, max_t=1000, solved_score=0.5, consec_episodes=100, print_every=1, train_mode=True):
    """Deep Deterministic Policy Gradient (DDPG)

    Params
    ======
        n_episodes (int)      : maximum number of training episodes
        max_t (int)           : maximum number of timesteps per episode
        train_mode (bool)     : if 'True' set environment to training mode
        solved_score (float)  : min avg score over consecutive episodes
        consec_episodes (int) : number of consecutive episodes used to calculate score
        print_every (int)     : interval to display results

    """
    memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, 2)
    agents = [
        Agent(base_dir + '/agent1.pytorch', memory, state_size=state_size, action_size=action_size, random_seed=2),
        Agent(base_dir + '/agent2.pytorch', memory, state_size=state_size, action_size=action_size, random_seed=2)
    ]

    mean_scores = []                               # list of mean scores from each episode
    min_scores = []                                # list of lowest scores from each episode
    max_scores = []                                # list of highest scores from each episode
    best_score = -np.inf
    scores_window = deque(maxlen=consec_episodes)  # mean scores from most recent episodes
    moving_avgs = []                               # list of moving averages

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name] # reset environment
        states = env_info.vector_observations                   # get current state for each agent
        scores = np.zeros(n_agents)                           # initialize score for each agent
        for agent in agents:
            agent.reset()
        start_time = time.time()
        for t in range(max_t):
            actions = [
                agent.act(states[i, np.newaxis], add_noise=True)
                for i, agent in enumerate(agents)
            ]
            env_info = env.step(actions)[brain_name]            # send actions to environment
            next_states = env_info.vector_observations          # get next state
            rewards = env_info.rewards                          # get reward
            dones = env_info.local_done                         # see if episode has finished
            # save experience to replay buffer, perform learning step at defined interval
            for agent, state, action, reward, next_state, done in zip(agents, states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done, t)
            states = next_states
            scores += rewards
            if np.any(dones):                                   # exit loop when episode ends
                break

        duration = time.time() - start_time
        min_scores.append(np.min(scores))             # save lowest score for a single agent
        max_scores.append(np.max(scores))             # save highest score for a single agent
        mean_scores.append(np.mean(scores))           # save mean score for the episode
        scores_window.append(max_scores[-1])         # save mean score to window
        moving_avgs.append(np.mean(scores_window))    # save moving average

        if i_episode % print_every == 0:
            print('\rEpisode {} ({} sec)  -- \tMin: {:.1f}\tMax: {:.1f}\tMean: {:.1f}\tMov. Avg: {:.1f}'.format(\
                  i_episode, round(duration), min_scores[-1], max_scores[-1], mean_scores[-1], moving_avgs[-1]))

        if moving_avgs[-1] >= solved_score and i_episode >= consec_episodes:
            print('\nEnvironment SOLVED in {} episodes!\tMoving Average ={:.1f} over last {} episodes'.format(\
                                    i_episode-consec_episodes, moving_avgs[-1], consec_episodes))
            for agent in agents:
                agent.save()
            break

    return mean_scores, moving_avgs

# run the training loop
mean_scores, moving_avgs = ddpg()
np.savez_compressed(base_dir + '/scores.npz', {
    'mean_scores': mean_scores,
    'moving_avgs': moving_avgs
})
env.close()
