# Continuous Reinforcement Learning

## Model-Based Approaches
- require a _known_ transition and reward model
- usually use dynamic programming iteratively
- examples
    - Policy iteration
    - Value iteration
    
## Model-Tree Approaches
- sample the environment, using exploratory actions, and use feedback/estimate to directly estimate value functions
- examples
    - Monte carlo methods
    - Temporal difference (TD) methods

## Deep Reinforcement Learning
- operates in continuous spaces (can't use a finite MDP)
- uses NNs 


## Discrete vs Continuous Spaces
- Discrete
    - State-value function (V): map of V -> R
        - can be represented by a dictionary
    - Action-value function (Q): SxA -> R
        - can be represented by a matrix
- - Continuous
    - The state and action spaces are all continuous/infinite
    - Two main approaches: *discretization* and *function approximation*
    - Starting simple with drastic discretization can yield quick prototypes of an algorithm, before moving on to (more complex) continuous approaches

## Tile Coding
- Can use non-uniform discretization where appropriate to improve performance
    - Tile coding: maintain several overlapping grids (tilings), and encode the continuous state as a bit vector indicating which tiles contain the current state
    - Adaptive tile coding: start with a coarse tiling, and iteratively split sub-tiles when a projected increase in performance is apparent. This removes the need to manually specify the tilings.
    - Coarse coding: use smaller, overlapping circles as the tilings, with a binary indicator for each circle containing the state. This yields a sparse, binary state encoding.
        - smaller circles -> less generalization, more resolution
        - larger circles -> more generalization, more resolution, smoother value functions
        - can use ellipses instead of circles to change the resolution<->generalization tradeoff in each dimension
        - instead of binary representation, can use the distance from each circle to reach a continuous, euclidean encoding (RBF)

## Function Approximation

## Discretization vs Function Approximation
- complex state space --> more discretization needed for performance, so either your data gets huge or you lose accuracy by keeping your data small
    - as the discretization error increases, the encoded positions change more erratically, and not in a natural, smooth way
