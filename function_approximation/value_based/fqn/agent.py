import torch
from torch import nn
from torch import optim
from torch.nn import MSELoss
import numpy as np 
import gymnasium as gym 
from gymnasium.spaces import Box, Discrete






class FittedQNet() :
    def __init__(self,
                 Env,
                 value_model_fn,
                 value_optimizer_fn,
                 value_optimizer_lr:float,
                 training_strategy_fn,
                 batch_size:int=32) :

        self.Env = Env

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
        print(f"number of states S = {self.nS}\nnumber of actions A = {self.nA}")

        
        #self.value_model_fn       = value_model_fn
        #self.value_optimizer_fn   = value_optimizer_fn
        #self.value_optimizer_lr   = value_optimizer_lr 
        #self.training_strategy_fn = training_strategy_fn
        
        #self.evaluation_strategy_fn= evaluation_strategy_fn
        #self.batch_size=batch_size

        self.cummulative_reward_per_episdode = 0
        
        self.online_model = value_model_fn(self.nS, self.nA)
        self.optimizer    = value_optimizer_fn(self.online_model,0.1)
        self.training_strategy_fn = training_strategy_fn
        self.value_optimizer_lr   = value_optimizer_lr
        self.batch_size=  batch_size
        
    def iteration_step(self, state) :
        
        action = self.training_strategy_fn.select_action(self.online_model,state)
        new_state, reward, terminal, truncated, _ = self.Env.step(action=action)
        is_terminated = terminal or truncated 
        experience = (state, action, reward, new_state, float(is_terminated))
        
        return new_state, is_terminated, experience 

        
    def train(self, max_episodes:int=1) :
        self.online_model = self.value_model_fn(self.nS, self.nA)
        self.optimizer    = self.value_optimizer_fn(self.online_model,0.1)
        self.experience =[]
        
        state, _ = self.Env.reset()
        for _ in range(max_episodes) :

            self.experience = []

            while True :
                state, is_terminated, experience = self.iteration_step(state=state)
                self.experience.append(experience)
                if is_terminated :
                    break
    

class FittedAgent :
    def __init__(self,
                 env_name:str,
                 value_model_fn,
                 value_optimizer_fn,
                 value_optimizer_lr:float,
                 training_strategy_fn,
                 batch_size:int=32,
                 max_buffer_size:int=1000) :
        
        self.env_name = env_name
        print(f"Selected env = {self.env_name}")        
        self.Env = gym.make(id=self.env_name,render_mode="human") 
        
        self.QNet = FittedQNet(Env=self.Env,
                               value_model_fn=value_model_fn,
                               value_optimizer_fn=value_optimizer_fn,
                               value_optimizer_lr=value_optimizer_lr,
                               training_strategy_fn=training_strategy_fn)
        self.batch_size =batch_size
        self.warmup = 0
        
    def act(self,state) :
        return self.QNet.iteration_step(state=state)
        
            
    def store(self,experience) :
        self.buffer.append(experience)
    
    
    def update(self) :
        pass
        
    def interact(self):

        state, _ = self.Env.reset()
        self.buffer = []
        while True :
            
            state, is_terminated, _ = self.act(state=state)
            
            if is_terminated :
                break

        
    
