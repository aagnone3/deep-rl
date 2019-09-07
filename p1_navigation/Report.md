# Project 1: Value-Based Learning for Navigation (and bananas!)

The following details my solution to training a reinforcement learning agent to navigate the banana environment using Deep Q Learning.

## Learning Algorithm
This solution employs the Deep Q learning algorithm, which employs a neural network to estimate the action-values for a given state in the environment. In particular, the Double DQN modification was used, which utilizes a 'target network' to estimate the action-values, while allowing the higher-variance 'local network' to still choose the highest-value action to take.

### Hyperparameters
Note: While there may be more hyperparameters than those listed below, these parameters are those that are of the utmost interest and effect on learning.

*Epsilon-greediness*: epsilon: 1->0.005, iteratively multiplied by 0.995 each episode
*Moving average rewards*: last 100 rewards tracked for performance measurement
*Memory replay buffer*: 100,000 experience tuples
*Memory replay batch size*: 64 experience tuples
*Discount factor*: 0.99
*Network weights soft update ratio*: 0.001 of local network weights used in target network
*Network learning rate*: 5e-4
*Episodes per learning update*: 4

### Neural Network Architecture(s)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1               [-1, 32, 35]             128
         MaxPool1d-2               [-1, 32, 18]               0
       BatchNorm1d-3               [-1, 32, 18]              64
            Conv1d-4               [-1, 16, 18]           1,552
         MaxPool1d-5                [-1, 16, 9]               0
       BatchNorm1d-6                [-1, 16, 9]              32
            Linear-7                   [-1, 64]           9,280
            Linear-8                    [-1, 4]             260
================================================================
Total params: 11,316
Trainable params: 11,316
Non-trainable params: 0
----------------------------------------------------------------

## Results
![rewards][rewards.png]

## Future Work
- Use various priority-defined sampling distributions over the memory replay buffer. A simple one is presented in the paper, but there is a lot of other potential for estimating the importance of experience tuples.
- Explore other mechanisms by which a 'target network' or 'teacher network' can lead and constrain the higher-variance local network.
