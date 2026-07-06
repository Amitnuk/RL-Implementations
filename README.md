# First Things First 
This repository is under active development. Some sections contain high-level intuition only and will later be formalized with mathematical detail and citations.
Grammatical mistakes, inconsistencies, and informal explanations are to be expected.

# Scope of This Repository:

As Richard P. Feynman once said, *"What I cannot create, I do not understand."* This repository reflects that philosophy through the implementation of reinforcement learning algorithms from scratch. Beyond studying the theory, the goal is to develop a deeper understanding of the underlying concepts by translating them into working code, while identifying and correcting gaps in my own understanding along the way.

This repository covers RL algorithms ranging from :
* *Tabular representation.*
* *Function approximation.*

# Environments and Development Framework: 
All the environments used in this repo are part of [Gymnasium](https://gymnasium.farama.org/index.html)  
As for the development frameworwork, I use PyTorch and tensorboard to experiment tracking.


# TODO 
- [X] [Neural Fitted Q Iteration](https://github.com/Amitnuk/RL-Implementations/tree/main/function_approximation/value_based/nfqi)
- [ ] [DQN](https://arxiv.org/abs/1312.5602)
- [ ] [Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [ ] [Reinforce](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- [ ] [DDPG](https://arxiv.org/pdf/1509.02971)
- [ ] [TD3](https://arxiv.org/abs/1802.09477)
- [ ] [SAC](https://arxiv.org/abs/1801.01290)
- [ ] [TRPO](https://arxiv.org/abs/1502.05477)
- [ ] [PPO](https://arxiv.org/abs/1707.06347)
- [ ] SARSA, Q Learning, etc
- [X] Repository structure
- [ ] Requirements to reproduce this repo results
- [ ] And more


# Repository Structure(Future)

```text
RL-Implementations/
│
├── tabular/
│   ├── dynamic_programming/
│   ├── q_learning/
│   └── sarsa/
│
├── deep_rl/
│   ├── value_based/
│   │   ├── nfqi/
│   │   ├── dqn/
│   │   └── double_dqn/
│   │
│   ├── policy_gradient/
│   │   └── reinforce/
│   │
│   └── actor_critic/
│       ├── on_policy/
│       │   ├── ppo/
│       │   └── trpo/
│       │
│       └── off_policy/
│           ├── ddpg/
│           ├── td3/
│           └── sac/
│
├── common/
│   ├── networks/
│   ├── buffers/
│   ├── policies/
│   └── utils/
│
└── experiments/
    ├── logs/
    │   ├── nfqi/
    │   ├── dqn/
    │   ├── double_dqn/
    │   ├── ddpg/
    │   ├── td3/
    │   ├── reinforce/
    │   ├── sac/
    │   ├── ppo/
    │   └── trpo/
    │
    └── plots/
        ├── nfqi/
        ├── dqn/
        ├── double_dqn/
        ├── ddpg/
        ├── td3/
        ├── reinforce/
        ├── sac/
        ├── ppo/
        └── trpo/
```
        

# Reinforcement Learning : 

**Reinforcement Learning**, a *Machine Learning* paradigm where an agent learns to make sequential decisions through trial and error within a given environment. 

# Objective 
The **Reinforcement Learning** goal is to find an **optimal policy**. A *policy* can be thought intuitively as the brain of the agent, where, for a given state, the policy ouputs the the action that better helps the agents reach its own goal. The agent goal is to maximize the long term reward.

Broadly, *RL* methods can be categorized into:
* *Value-based methods*
* *Policy-based methods*
* *Model-based methods*

This repository only covers *Value-based methods* and *Policy-based methods*

