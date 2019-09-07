# Deep Q Learning

Using NNs to approximate the value function tends to (by itself) cause erratic and non-smooth values.
The Deep Q-learning algo addresses these instabilities through two key features:
- Experience replay
    - each experience now has the potential to influence several weight updates, resulting in output smoothing
- Fixed Q-targets
    - a target network represents the old Q function, and is used to compute the loss of every action during training
    - always try to predict the old Q value, since the Q values are always changing. This results in the estimated Q values not spiraling out of control

## Experience Replay
- store experiences after learning from them. Keep them in a replay buffer, and keep sampling from them during learning.
- can make sure you recall rare events
- to prevent the network from getting stuck learning only from state<-> action correlations present in the _sequence_ of examples, sample randomly from the replay buffer
    - ex: tennis, states are what side the ball is coming on, actions are forehand or backhand
        - if I don't have experience replay, I can get stuck hitting forehand -> ball comes back to right side -> hit forehand etc
            - the estimated action-value then suggests always hitting forehand, since that is where all the samples are
        - incorporating experience replay, I make sure that I am re-visiting other states as I learn from newer samples

## Fixed Q-Targets
The TD target represents the _true_ q_{\pi}(S,A) that we are trying to learn, which is independent of any parametrization (i.e. weights) of it.
Since the vanilla TD target definition includes w, it creates a correlation betweeh the NN target and its parametrization.
This is updating a guess with a guess, and is chasing a moving target.

What needs to happen is de-coupling the target from the action (don't train the donkey to walk straight while _riding the donkey_ with the carrot in a stick, be in front of the donkey, walking backwards).


## References
[Ref Site: Deep Q Learning](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_q_learning.html)
[DQL Paper](http://files.davidqiu.com//research/nature14236.pdf)
[Thrun, Schwartz: Issues in using function approx...](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.3097&rep=rep1&type=pdf)
[Riedmiller: Neural Fitted Q Iteration](http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf)
