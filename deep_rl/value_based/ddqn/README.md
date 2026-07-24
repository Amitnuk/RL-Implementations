# First Things First 
This repository is under active development. Some sections contain high-level intuition only and will later be formalized with mathematical detail and citations.
Grammatical mistakes, inconsistencies, and informal explanations are to be expected.

# Goal 
Solving CartPole, Lunar Lander, mountain car, acrobot and the Atari Games using [Double DQN](https://arxiv.org/abs/1509.06461).

These environments are part of [Gymnasium](https://gymnasium.farama.org/index.html), a quick overview can be found at [[1](https://github.com/Amitnuk/RL-Implementations/tree/main)]

# Double DQN

## Introduction

### Motivation 
The core ideas behind DDQN are the same as those of DQN, where the i.i.d issue is mitigated with replay buffer and non stationarity of the bootstrapped issue is mitigated with a target network.  As stated in [2](https://github.com/Amitnuk/RL-Implementations/tree/main/deep_rl/value_based/dqn), DQN tends to overestimates the Q values, due to the maximization operator. In the original DQN, the action selection and evaluation of Q are correlated, while DDQN decorrelates them.


### Background

#### Double Q learning 


$$
Q_{\phi(A)}(s,a) \leftarrow r + \gamma Q_{\phi B}(s',\arg\max_{a'} Q_{\phi A}(s', a'))
$$ 

$$
Q_{\phi(B)}(s,a) \leftarrow  r + \gamma Q_{\phi A}(s',\arg\max_{a'} Q_{\phi B}(s', a'))
$$

The idea is to use two networks, one that selects the action while the other evalutates Q. This reduces the correlation between the action selection and the Q value action evaluation.

Fortunatily, we already have two network, the target network and the main network, which we can use without adding any overhead. 

The main network selects the action :

$$
a^{\ast}  = \arg\max_{a'} Q_{\phi}(s', a'))
$$

The target network evaluates the action :

$$
Q_{\phi^-}(s',a^{\ast})
$$


Together, this gives:

$$
y = r + \gamma Q_{\phi^-}(s', \arg\max_{a'} Q_{\phi}(s', a'))
$$


In double q learning, both networks alternate roles during training. DDQN has a dedicated target network so this switching is not necessary.


# Usage 
## Training


## Testing


# TODO

- [X] train module
- [X] training cartpole 
- [ ] training lundar lander
- [ ] training mountain car
- [ ] training acrobot
- [ ] testing module
- [ ] plots
- [X] explanation
- [ ] requirement and a usage section
- [ ] grammar correction and some other minor details




# Empirical results and training configurations
## Cartpole Just a placeholder
<table>
  <tr>
    <td align="center">
      <b>Cartpole reward and average reward curve</b>
    </td>
    <td align="center">
      <b>Animation</b>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/plots/reward_curve_cartpole.png?raw=true" width="400"/>
    </td>
    <td>
      <img src="https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/figures/dqn_cartpole.gif?raw=true" width="400"/>
    </td>
  </tr>
</table>



## Lunar lander Just a placeholder

<table>
  <tr>
    <td align="center">
      <b>Lunar lander reward and average reward curve</b>
    </td>
    <td align="center">
      <b>Animation</b>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/plots/reward_curve_lunarlander.png?raw=true" width="400"/>
    </td>
    <td>
      <img src="https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/figures/dqn_lunarlander.gif?raw=true" width="400"/>
    </td>
  </tr>
</table>




## Mountain car Just a placeholder

<table>
  <tr>
    <td align="center">
      <b>Mountain reward and average reward curve</b>
    </td>
    <td align="center">
      <b>Animation</b>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/plots/reward_curve_mountaincar.png?raw=true" width="400"/>
    </td>
    <td>
      <img src="https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/figures/dnq_mountaincar.gif?raw=true" width="400"/>
    </td>
  </tr>
</table>





## Acrobot Just a placeholder


<table>
  <tr>
    <td align="center">
      <b>Acrobot reward and average reward curve</b>
    </td>
    <td align="center">
      <b>Animation</b>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/plots/reward_curve_acrobot.png?raw=true" width="400"/>
    </td>
    <td>
      <img src="https://github.com/Amitnuk/RL-Implementations/blob/main/experiments/figures/dqn_acrobot.gif?raw=true" width="400"/>
    </td>
  </tr>
</table>






# Summary 


# References

[CS 285 : Deep RL, 2023](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps).

[Sutton, R.S., Bach, F., and Barto, A.G., 2018. Reinforcement learning: An introduction. Massachusetts: MIT Press Ltd.](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)

[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

[Mathematical Foundations of Reinforcement Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)

[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)




