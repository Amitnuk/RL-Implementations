# First Things First 
This repository is under active development. Some sections contain high-level intuition only and will later be formalized with mathematical detail and citations.
Grammatical mistakes, inconsistencies, and informal explanations are to be expected.

# Goal 
Solving CartPole, Lunar Lander, and Mountain Car Using [Neural Fitted Q-iteration](https://link.springer.com/chapter/10.1007/11564096_32) 

## Problem definition 

In classical reinforcement learning, we normally use tables, which can be viewed as a non-parametric representation. One of the key requirements for convergence in this setting is that each state or state-action pair must be visited infinitely often. This is normally achieved through exploration.

One of the issues with using tables is their limited capacity. Creating a table over a high-dimensional space, a continuous space, or a combination of both makes the problem intractable. Although one could discretize the space, *but nah, we ain't doing that, because FNQ, duh.* On a serious note, a common alternative to tabular methods is to use parametric function approximators, which provide generalization across the state space, and the function does not need to be linear.

A parametric function approximator allows the agent to learn from a limited number of samples without requiring (or even being able to achieve) infinite visitation of every state. This becomes especially important in continuous spaces, where exhaustive coverage is infeasible. However, generalization in reinforcement learning is a double-edged sword. While it enables sharing information across similar states, it also introduces coupling through shared parameters, meaning that updating the value function in one region of the state space can negatively affect other regions.

As a result, classical convergence guarantees may no longer hold when using function approximators, and learning dynamics may become unstable or even diverge depending on the function class, the algorithm, and the data distribution. And this is where [Fitted Neural Q-iteration](https://link.springer.com/chapter/10.1007/11564096_32) comes in. 

## Neural Fitted Q-iteration

Multiple approaches have tried to leverage the power of neural networks by representing the value function as multi-layer perceptrons, but the results were not satisfactory, the reason being what we already discussed. While generalization is desirable, it can undo the effort made in one region of the state space based on a weight update in another region, resulting in divergence, because the system never settles.

* How can we exploit the positive properties of global approximators while avoiding the negative ones ?

The novel idea was that, for each data point used during an update, previous knowledge must be available, effectively transitioning from an online approach to an offline one.
They implement this idea by storing all previous experiences in terms of state-action transitions in memory.

For interessed reader i recommend the [Neural Fitted Q-iteration](https://link.springer.com/chapter/10.1007/), but overall what the algorithms does is :
* 1. create the dataset of transitions using some policy
* 2. create the bootstrapped target 
* 3. minimize the error 

where steps 2 and 3 are repeated *k times*, meaning we freeze the target network and then perform batch gradient descent. Oversimplifying each step reduces the problem to a fully static supervised learning problem.



## My Own Ramblings 

Yeah, that was it; all they did was do batch gradient descent instead of stochastic gradient descent by keeping the transitions in a buffer. Then the [DQN](https://arxiv.org/abs/1711.07478) creator was like, “euh, what if we kept the transition for longer, and we kept replaying it even after?”

All jokes aside, my assumption for [DQN](https://arxiv.org/abs/1711.07478) creation is most likely wrong. Although both algorithms have the same goal—solving the Bellman optimal equation—both approaches solve the problem differently.

[DQN](https://arxiv.org/abs/1711.07478) does not transform the problem into a supervised learning (regression) problem; in its essence, it is more closely related to [NFQI](https://link.springer.com/chapter/10.1007/. More the details here [In contrustion]()


I am going a bit off the rails here, but if you ignore the policy gradient part in the actor-critic approach, one can derive fitted Q-iteration. If you are interested, see [Video at 11:05](https://www.youtube.com/watch?v=QUbuBEY12u0&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=27). Ici c'est Paris, i mean [NFQI](https://link.springer.com/chapter/10.1007/11564096_32).

Let us move move on

### CartPole 

### Lunar Lander 
The Lunar Lander environment simulates landing a small rocket on the moon surface. 
The environment is available in [Gymnasium](https://gymnasium.farama.org) 

#### Background 
As with most (if not all) [Gymnasium](https://gymnasium.farama.org) environments is assumed to be an [MDP](https://en.wikipedia.org/wiki/Markov_decision_process) which is remains unknown to the agent. As consequence, it requires us to interact with the environment to samples transitions through agent–environment interaction. 

The agent starts by receiving an initial observation, selects an action, and sends it to the environment. The environment returns a reward and the next observation, and this process repeats until a termination condition is reached. 

In [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander) the observation space is an 8-dimensional continuous vector and the action space is reresented with 4 discrete action ranging for 0 to 3. The observation space is a lower dimensional space, but being continuous makes it hard for classic reinforcement learning, one could discretize the observation space which has is own issues but a better approach is to use parametric function approximator instead of non parametric one. 



### Mountain Car

# Usage 
## Training

## Testing

# TODO
- [X] train module
- [X] training cartpole 
- [X] training lundar lander
- [X] training mountain car
- [X] training acrobot
- [ ] testing module
- [ ] conclusions
- [ ] plots
- [ ] a more rigorous explanation
- [ ] And more



# Notes

# Results

![Alt text](https://github.com/Amitnuk/RL-Implementations/blob/main/function_approximation/value_based/nfqi/results/gifs/cartpole.gif)





