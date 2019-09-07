# Policy-Based Methods

Use an NN to directly estimate the action, instead of just the action-value.

*Stochastic policy search*: estimating the optimal policy by randomly perturbing the most recent best estimate of the NN weights

Here, we're going black-box optimization, which does not maximize expected return!

## Cross-Entropy Method
Select the top-k and average

## Evolution Strategies
Use a weighted average

## Why Policy-Based Methods?
- Simplicity
    - bypass estimating the optimal value function (is it necessary?)
    - \pi(s,a) = P[a|s]
    - this is more direct, and saves a lot of intermediate {steps,data} compare to value-based methods
- Stochastic Policies
    - epsilon-greedy policies are only a crude sense of a stochastic policies. Policy-based methods learn a more true stochastic policy
    - when aliased states (2 states that appear to be 1) occur, a less-stochastic value-based policy will have an identical mapping for both states (save for a little bit of epsilon-greedy helping). With a more stochastic policy, the only thing that is identical is the _probability_ of the actions. Since it's only a probability, the agent will likely "get out" of the rut of having identical probability mappings for the two states
- Continuous Action Spaces
    - in continuous action spaces, finding the maximum-value action repeatedly is very expensive, whether discretizing or parametrizing. With action-value methods, you bypass the valuation step, and approximate the policy directly.
