# Monte Carlo Control

## Two steps (back-and-forth)
- Policy evaluation
    - collect an episode with the current \pi estimate
    - update the Q-table, which is the value estimate for each {state, action} pair
        - Q(S_t,A_t) = Q(S_t,A_t) + \alpha(G_t - Q(S_t,A_t))
            - G_t is the current return (at time step t)
            - when the current return does not match the Q-table estimate, G_t - Q(S_t,A_t) != 0, and we update the Q-table estimate to be closer to the current return. The extend to which we update is controlled by the \alpha hyperparamter.
        - G_t = \sum_{s=t+1}^T \gamma^{s-t-1} R_s
            - gamma: the discount rate
                - gamma=0: no memory of the past, always follow the present
                - gamma=1: full memory of the past
- Policy improvement
    - update the estimate of \pi as the \epsilon-greedy solution, according to the current Q-table
