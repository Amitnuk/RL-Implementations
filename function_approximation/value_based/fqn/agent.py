import torch
from torch import nn
from torch import optim
from torch.nn import MSELoss
import numpy as np 
import gymnasium as gym 
from gymnasium.spaces import Box, Discrete






class FittedQNet() :
    pass


class FittedAgent :
    def __init__(self,
                 env_name:str,
                 value_model_fn:nn.Module,
                 value_optimizer_fn,
                 value_optimizer_lr:float,
                 training_strategy_fn) :
        
        self.env_name = env_name
                
        self.Env = gym.make(id=self.env_name,render_mode="human") 
        _,_ = self.Env.reset()


        if isinstance(self.Env.action_space,Box) :
            self.action_dim = self.Env.action_space.shape[0]
        elif isinstance(self.Env.action_space,Discrete) :
            self.action_dim =self.Env.action_space.n
        else:
            raise NotImplementedError(f"Unsupported action space: {type(self.Env.action_space)}")

        if isinstance(self.Env.observation_space,Box) :
            self.observation_dim = self.Env.observation_space.shape[0]
        elif isinstance(self.Env.observation_space,Discrete) :
            self.observation_dim =self.Env.observation_space.n
        else:
            raise NotImplementedError(f"Unsupported onservation space: {type(self.Env.observation_space)}")
        
        self.nS, self.nA = self.observation_dim, self.action_dim
        print(f"Selected env = {self.env_name}\nnumber of states S = {self.nS}\nnumber of actions A = {self.nA}")

        #self.online_model = value_model_fn(self.nS, self.nA)


        self.value_model_fn       = value_model_fn
        self.value_optimizer_fn   = value_optimizer_fn
        self.value_optimizer_lr   = value_optimizer_lr 
        self.training_strategy_fn = training_strategy_fn
        
        #self.evaluation_strategy_fn= evaluation_strategy_fn
        #self.batch_size=batch_size

        self.cummulative_reward_per_episdode = 0
    def interact(self):

        while True :
            
            action = self.Env.action_space.sample()
            

            state, reward, terminal, truncated, info = self.Env.step(action=action)
            self.cummulative_reward_per_episdode += reward
            
            if truncated or terminal :
                break

        print(f"Cummulative = reward : {self.cummulative_reward_per_episdode}")   
