# First Things First 
This is repo is still in construction

# Goal 
Solving CartPole, Lunar Lander and Mountain Car using [Fitted Neural Q-iteration](https://link.springer.com/chapter/10.1007/11564096_32) 

## Problem definition 

In classic RL, we normally use tables, which can be viewed as a non-parametric representation and a key requirement fconvergence in this setting is that each state must be visited infinitely often.

One of the issues using tables is the capacity limitations, creating a table over a high-dimensional space, continuous space or a combination of both makes the problem intractable, although one could discretise the space, *but Nah, we ain't doing that, because FNQ duh. On a serious note*, a common alternative to the tabular method is to use parametric function approximators, which provide generalization across the state space and the function does not need to be linear. 

A parametric function approximator allow the agent to learn from a limited number of samples without requiring (or even being able to achieve) infinite visitation of every state. This becomes especially important in continuous spaces, where exhaustive coverage is infeasible. However, generalization in reinforcement learning is a double-edged sword. While it enables sharing information across similar states, it also introduces coupling through shared parameters, meaning that updating the value function at one region of the state space can negatively affect other regions. 

As a result, classical convergence guarantees may no longer hold when using function approximators, and learning dynamics may become unstable or even diverge depending on the function class, the algorithm, and the data distribution. And this is where [Fitted Neural Q-iteration](https://link.springer.com/chapter/10.1007/11564096_32) comes along. 

## Fitted Neural Q-iteration

Multiple approaches have tried to leverage the power of Neural Networks by representing value function as multi-layer perceptrons, but the results where not satisfatory the reason being what we already discussed. While generalization is desirable, it can destroy the effort done in some region of the state space based on the current weigth update in another region of the state space, resulting in a disvergence, because we never settle.

* How to exploit positive properties of global approximator while avoiding the negative ones ?

The nouvel idea was, for each single datapoint during a update a previous knowledge must be available. Effectively trasitioning from online approach to a offline one.
They implement this idea by storing all the previous experiences in terms of state-action  transitions in memory.


## My Own Ramblings 

Yeah, that was it, all they did was do batch gradient descent instead of stochastic gradient descent by keeping the transitions in a buffer. Then the [DQN](https://arxiv.org/abs/1711.07478) creator where like, euh what if we kept the transition for longer, and we kept replaying it even after ?
All jokes aside, i found the motivation of [FNQI](https://link.springer.com/chapter/10.1007/11564096_32) more insteristing starting from batch [Actor Critic Methods](https://proceedings.neurips.cc/paper_files/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf).

I am going bit off the rails here, but if you just ignore the policy gradient part on actor critic approach, one can derive fitted Q-iteration, if you are interested,  see [Video at 11:05](https://www.youtube.com/watch?v=QUbuBEY12u0&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=27). Ici c'est Paris, i mean [FNQI](https://link.springer.com/chapter/10.1007/11564096_32).

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
python3 function_approximation/value_based/fqn/train.py --env lunarlander --max_episodes 1000 --batch_size 1024  --gamma 1 --epsilon 0.5 --epochs 40 --optimizer 1 --lr 0.0003 --use_gpu
python3 function_approximation/value_based/fqn/train.py --env lunarlander --max_episodes 1000 --batch_size 1024  --gamma 1 --epsilon 0.5 --epochs 50 --optimizer 1 --lr 0.0003 --use_gpu

# TODO
- [X] train module
- [X] training cartpole 
- [ ] training lundar lander
- [ ] training mountain car
- [ ] training acrobot
- [ ] testing module
- [ ] restruncturing
- [ ] plots
- [ ] And more



# Notes

# Results

# More