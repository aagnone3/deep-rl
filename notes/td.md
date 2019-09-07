# Temporal Difference (TD) Learning

Monte carlo learning requires running multiple _full simulations_ of an environment, in order to extract sample statistics from those multiple runs. However, not all situatiosn are very amenable to full simulations, and many can also benefit from having "within-episode" notions of feedback to use to guide its behavior.

TD methods update the Q table after _every_ time step. Analogy: what SGD is for GD.
During the update, you use the Q table to estimate the action-value of the state you have just ended up in.

Sarsa(0) uses as its next {state,action} the 

sarsa
- take the action
- choose the next action
- update according to next action value

sarsa-max/Q learning
- choose the action
- take the action
- update according to action in next state that yields most

expected sarsa
- use Bayes to update according to the expected action-value of the next state, instead of blindly using the max. this protects the algorithm from being largely moved by unlikely, but high-magnitude, action values.

GLIE: Greedy in the limit, with infinite exploration
Conditions:
- \epsilon_i > 0 for all time steps i
- \epsilon_i decays to zero as i approaches \infty

An initialized Q-table is called "optimistic" if its initial action-value estimates are guaranteed to be larger than the true action values.
