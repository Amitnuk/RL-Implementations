# First Things First 
This repository is under active development. Some sections contain high-level intuition only and will later be formalized with mathematical detail and citations.
Grammatical mistakes, inconsistencies, and informal explanations are to be expected.

# Goal 
Solving CartPole, Lunar Lander, and many more using [DQN](https://arxiv.org/pdf/1312.5602).

These enviroments are part of [Gymnasium](https://gymnasium.farama.org/index.html), a quick overview can be found at [[1](https://github.com/Amitnuk/RL-Implementations/tree/main)]

# DQN

## Introduction

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
- [ ] plots
- [ ] a rigorous explanation
- [ ] requirement and a usage section
- [ ] grammar correction and some other minor details



# Notes

# Results
## Graphs


# Summary 



# References

[CS 285 : Deep RL, 2023](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps).
[Sutton, R.S., Bach, F., and Barto, A.G., 2018. Reinforcement learning: An introduction. Massachusetts: MIT Press Ltd.](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)


python3 deep_rl/value_based/dqn/main.py --use_gpu --launch train --env mountaincar --max_episodes 70000 --buffer_size 200000 --gamma 0.99 --behaviour_policy egreedy --batch_size 32 --optimizer 0 --lr 0.001251 --epsilon 1.0 --final_epsilon 0.00001 --decay_ratio 0.85 --seed 34

hidden_units = (24,48) 

works the best
python3 deep_rl/value_based/dqn/main.py --use_gpu --launch train --env mountaincar --max_episodes 7000 --buffer_size 500000 --gamma 0.99 --behaviour_policy egreedy --batch_size 32 --optimizer 1 --lr 0.0001251 --epsilon 0.5 --final_epsilon 0.1 --decay_ratio 0.2 --seed 34

hidden_units = (168,48)


best of the best
python3 deep_rl/value_based/dqn/main.py --use_gpu --launch train --env mountaincar --max_episodes 4000 --buffer_size 200000 --gamma 0.99 --behaviour_policy egreedy --batch_size 32 --optimizer 1 --lr 0.0001251 --epsilon 0.5 --final_epsilon 0.2 --decay_ratio 0.3 --seed 34

hidden_units = (64,64) 

if np.sum(Agent.episode_timestep) % 7000 == 0 :





