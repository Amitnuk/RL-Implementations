# First Things First 
This repository is under active development. Some sections contain high-level intuition only and will later be formalized with mathematical detail and citations.
Grammatical mistakes, inconsistencies, and informal explanations are to be expected.


# Reinforcement Learning : 

**Reinforcement Learning**, a *Machine Learning* paradigm where an agent learns to make sequential decisions through trial and error within a given environment. Sequential decision making can be formalized through a [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) (MDP) . 


# Markov Decision Process
An MDP is defined by the tuple **(S, A, r, P, γ, H)**.
Where:
*S* represents a set of possible states the agent can find itself in the environment.
*A* represents a set of possible actions an agent can take.
*r* represents a reward, a scalar signal given to the agent after an action is taken in a state.
*P* probability distribution **p(s'|s,a)**, describing the state the agent transitions to after taking an action in the environment.
*γ* is a discount factor that weights the importance of future rewards.
*H* is the horizon, that specifies the number of interactions allowed in an episode.


# Objective 
The agent's goal is to find an **optimal policy** which maximizes the *expected return* (cumulative discounted rewards). 

$$
\mathbb{E}\left[\sum_{t=0}^{H} \gamma^t r_t \right]
$$


# Core Method Families:
To reach this goal, the agent can leverage three different approaches, a *value-based*, *policy-based* and *model based*:


## Value-based:
The agent learns :
* state-value function *v(s)* : the value of being in a state.
* action-value function *q(s, a)*: the value of taking an action in a state.

## Policy-based:
The agent parameterizes and optimizes directly the optimal policy.

$$
\pi(a \mid s)
$$

## Model-based:
The agent learns the model of the environment.

# Representation of Value Functions and Policies:
* Tabular:
For lower-dimensional and discrete state/action space, value functions and the policy can be represented with a table(*which can be seen as a non-parametric function*).

* function approximation:
For high-dimensional or continuous state/action space, the tabular representation becomes infeasible. One could discretize the space; a more modern approach is to leverage parametric functions through function approximation. 

What makes function approximation attractive is *generalization*, which eliminates the need of infinitely visiting a state often, as stated in GLIE (Greedy in the Limit with Infinite Exploration). In similar states, the agent should behave in the same manner, but using function approximation comes with a cost: the convergence of the policy to a greedy policy may not hold anymore. 


# Scope of This Repository:
This repo covers RL algorithms ranging from :
* tabular representation
* function approximation.

# Evironments: 
All the environment used in this repo are part of [Gymnasium](https://gymnasium.farama.org/index.html)  


# TODO 
- [X] [Fitted Neural Q Iteration](https://github.com/Amitnuk/RL-Implementations/tree/main/function_approximation/value_based/nfqi)
- [ ] [DQN](https://arxiv.org/abs/1312.5602)
- [ ] [Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [ ] [Reinforce](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- [ ] [SAC](https://arxiv.org/abs/1801.01290)
- [ ] Tabular : SARSA, Q Learning, etc
- [ ] And more

