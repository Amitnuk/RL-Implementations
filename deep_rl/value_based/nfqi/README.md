# First Things First 
This repository is under active development. Some sections contain high-level intuition only and will later be formalized with mathematical detail and citations.
Grammatical mistakes, inconsistencies, and informal explanations are to be expected.

# Goal 
Solving CartPole, Lunar Lander, and Mountain Car Using [Neural Fitted Q-iteration](https://link.springer.com/chapter/10.1007/11564096_32) 
These enviroments are part of [Gymnasium](https://gymnasium.farama.org/index.html), a quick overview can be found at [1](https://github.com/Amitnuk/RL-Implementations/tree/main)

# Neural Fitted Q-iteration

## Introduction

NFQI, a special case of *Fitted Q-iteration* algorithms, this algorithm mitigates the problem that arise when using multi-layer perceptron as function approximator. 
Function approximator, such MLP, works on a global level which helps leveraging *generalization*. Althought, *generalization* is desirable property, it comes with a caveat in *RL*. Since weight parameters are coupled and the Bellman backups are use instead, in absence of true labels, to guide the learning, as a consequence *updating a weight in a given region of the state space can impact negatively in other reagions. and thus destroying the all the progress done in other regions.* 

Function approximator, a part of the deadly triad in *RL*, alleviate the conditions in *GLIE*. However the convergence guarantees may not hold.


### Main idea 

### Bellman operator
NFQI, its a value iteration algorithm in its core,  where the Bellman operator *T* is applied to the action value *Q(s,a)* continously still reaching the fixed point

$$
  Q \leftarrow (\mathcal{T}Q)
$$

To prove that the convergence to the fixed point is guaranteed it suffice to show that the Bellman operator is a [contraction](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem).

### Fitting operator
After applying the bellman operator we regress the model to the Bellman backups, also known a projection because it project the belman backup back in the set defined by the function approximation. This steps is essentially  defines another operator

$$
\Pi := (\mathcal{T}\Pi)
$$

One can akso show that $\Pi$ is also a contraction
### Fitted value iteration algorithm 
combining the both opoerator we get :

$$
Q_{k+1} \leftarrow \Pi (\mathcal{T}Q_k)
$$

Despite both operator being contractions, the combination of both is not a contraction because both operators are contraction under two different norms. The Belman  operator being under max norm and the fitting oprator under L2 norm.
Intuitively, applying the bellman operator makes us close to fixed point and applying fitting operator projects the bellman backup back to the set defined by function approximator, which is far from the fixed point. Thus, fitted iteration does not converge in general.



## Algorithm 
$$
(1)\quad collect \mathcal{D} = \{(s_i, a_i, s'_i, r_i)\}_{i=1}^N
$$

$$
(2)\quad y_i = r_i + \gamma \max_{a'} Q_\phi(s'_i, a')
$$

$$
(3)\quad \phi \leftarrow \arg\min_\phi \sum_{i=1}^N \left( Q_\phi(s_i, a_i) - y_i \right)^2
$$

where (2) and (3) are repeated *k* times. Step (2) give us fixed bellman backups and step (3) minimizes the bellman error.


# Usage 
## Training

## Testing

# TODO
- [X] train module
- [X] training cartpole 
- [X] training lundar lander
- [X] training mountain car
- [X] training acrobot
- [X] testing module
- [X] plots
- [X] a more rigorous explanation
- [ ] grammar correction and some other minor details



# Notes

# Results
## Graphs
### Cartpole  

| Moving Average Reward(Eval) | Total Average Reward(Eval) |
|----------|----------|
| ![Moving Average Reward(Eval)](https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/plots/cartpoleaveragereward1000.png) | ![Total Average Reward](https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/plots/cartpolereward.png) |

The graphics above showcase the total reward and the average reward using 1000 last collected reward in evaluation, as we can see the agent learns to maintain the pole upright. The Q-value is model by MLP with two hidden layers with 512 and 128 neurons respectivelly, with a batch size of 1024 and discount factor 1. The activation function is a ReLU and optimizer is RMSProp with a learning rate of 0.0003. The behaviour policy is epsilon greedy with a decaying epsilon that starts from 1.0 and stops at 0.5. The result of the agent managing to maintain the pole uprigth can be seen in gifs in the animation subsection.


### Lunar lander  

| Moving Average Reward(Eval) | Total Average Reward(Eval) |
|----------|----------|
| ![Moving Average Reward(Eval)](https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/plots/lunarlanderaveragereward1000.png) | ![Total Average Reward](https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/plots/lunarlanderreward.png) |


The graphics above showcase the total reward and the average reward using 1000 last collected reward in evaluation, as we can see the agent learns to land the rocket. The Q-value is model by MLP with two hidden layers with 256 and 256 neurons respectivelly, with a batch size of 1024 and discount factor 0.99. The activation function is a ReLU and optimizer is RMSProp with a learning rate of 0.0003. The behaviour policy is epsilon greedy with a decaying epsilon that starts from 1.0 and stops at 0.1. The result of the agent managing to maintain the pole uprigth can be seen in gifs in the animation subsection.


### Mountain Car
Throughout the training, the reward remained at -200, indicating the agent did not learn the policy to escape the valley.

## Animation 
### Lunar lander
![Cartpole](https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/figures/cartpole.gif)


### Lunar lander
![Lunar Lander](https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/figures/lunarlander.gif)



# Summary 

NFQI is far from a state of the art, but in simple environment like lunarlander and cartopole, it shows a good performance as we can see on the grapths above, but in environment like mountaincar where the reward is uninformatively dense, an environment that does not provide directional information to guide the agent, NFQI, struggled. A solution I found to solve mountain car involved reward shapping based on the position to the goal.

Overall, we showed that NFQI algotihm converges to a good policy, but it has two glaring issues. First, the fact we have moving target, in each update, we have a different the Bellman backup. Thus, in each update the agent solves a different regression problem. Second, the algorithm, as stated in the paper, is not sample efficient, the data collected is thown away before each update and a new one must be collected. Both problems are tackled in [DQN](https://arxiv.org/abs/1312.5602), an algorithm that I also implement in this repository.



# Reference 

[CS 285 : Deep RL, 2023](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps)
[Sutton, R.S., Bach, F., and Barto, A.G., 2018. Reinforcement learning: An introduction. Massachusetts: MIT Press Ltd.](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)







