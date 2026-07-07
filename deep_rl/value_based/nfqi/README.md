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
- [ ] conclusions
- [ ] plots
- [ ] a more rigorous explanation
- [ ] And more



# Notes

# Results
## Graphs
### Cartpole  

| Moving Average Reward(Eval) | Total Average Reward(Eval) |
|----------|----------|
| ![Moving Average Reward(Eval)](https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/plots/cartpoleaveragereward1000.png) | ![Total Average Reward](https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/plots/cartpolereward.png) |

### Lunar lander  

| Moving Average Reward(Eval) | Total Average Reward(Eval) |
|----------|----------|
| ![Moving Average Reward(Eval)](https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/plots/lunarlanderaveragereward1000.png) | ![Total Average Reward](https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/plots/lunarlanderreward.png) |

## Figures 
### Lunar lander
![Cartpole](https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/figures/cartpole.gif)


### Lunar lander
![Lunar Lander](https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/figures/lunarlander.gif)








