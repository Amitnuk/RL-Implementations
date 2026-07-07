import torch
from torch import nn
from torch import optim
from torch.nn import MSELoss
import numpy as np 
import gymnasium as gym 
from gymnasium.spaces import Box, Discrete
from common.buffers.replaybuffer import ReplayBuffer
import random
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from network import FCQNetwork


class DQNAgent :
    def __init__(self,
                 env_name:str,
                 value_model_fn,
                 value_optimizer_fn,
                 value_optimizer_lr:float,
                 training_strategy_fn,
                 gamma:float=0.99,
                 batch_size:int=100,
                 epochs:int=40,
                 seed:int=34,
                 mode:str=eval) :
        
        self.mode = mode
        self.env_name = env_name
       
     
        if self.mode == "eval" :
            self.Env = gym.make(id=self.env_name, render_mode="human")
            
        
        if self.mode == "train" :
            self.Env = gym.make(id=self.env_name)
            
        print(f"Selected env = {self.env_name}")        

        torch.manual_seed(seed=seed)
        torch.cuda.manual_seed(seed=seed)
        np.random.seed(seed=seed)
        random.seed(seed)

        self.Env.action_space.seed(seed=seed)

        if isinstance(self.Env.action_space,Discrete) :
            self.action_dim =self.Env.action_space.n
            self.low  = 0
            self.high = int(self.action_dim) - 1
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
        print(f"Single State = {self.Env.observation_space.sample()}\nSingle Action = {self.Env.action_space.sample()}")
        
        self.cummulative_reward_per_episdode = 0

        
        self.model:FCQNetwork = value_model_fn(self.nS, self.nA)
        
        self.value_optimizer_lr   = value_optimizer_lr
        self.optimizer    = value_optimizer_fn(self.model,self.value_optimizer_lr)
        self.training_strategy_fn = training_strategy_fn
        self.batch_size=  batch_size
        
        self.seed = seed
        self.gamma = gamma
        self.batch_size =batch_size

        self.epochs = epochs
        self.buffer = ReplayBuffer(batch_size=self.batch_size)
        self.learning_rate = value_optimizer_lr
        self.loss = 0
        self.training_strategy_fn = training_strategy_fn
        
        print("DQNAgent")


    def act(self,state) :
        return self.iteration_step(state=state)
        
    def load(self):
        return self.buffer.load()
    
    def store(self,experience) :
        self.buffer.store(experience=experience)


    def clear(self) :
         self.buffer.clear()

    def iteration_step(self, state) :
        
        action = self.training_strategy_fn.select_discret_action(self.model,state)
        new_state, reward, terminal, truncated, _ = self.Env.step(action=action)
        is_terminated = terminal or truncated
        experience = (state, action, reward, new_state, float(terminal))

        self.store(experience=experience)
        return new_state, is_terminated 
    
    def update(self, experiences) :
        states, actions, rewards, next_states, is_terminated = experiences
        
        q_sa     = self.model(states).gather(1, actions)
        
        max_a_q_sp = self.model(next_states).detach().max(1)[0].unsqueeze(1)
        td_target_qs = rewards +  self.gamma * max_a_q_sp * ( 1 - is_terminated )
        
        td_errors  = td_target_qs - q_sa

        value_losses = td_errors.pow(2).mul(0.5).mean()

        self.loss = value_losses.detach().cpu().numpy()
 
        self.optimizer.zero_grad()
        value_losses.backward()
        self.optimizer.step()

    def evaluate(self, max_episodes:int=1):

        rewards=[]
        state, _ = self.Env.reset()
        
        for i in range( max_episodes ) :
            if self.mode == "eval" :
                print(f"episode[{i+1}/{max_episodes}]")
            rewards.append(0.0)
            while True :
                
                with torch.inference_mode() :
                    q_values = self.model(state).detach().cpu().data.numpy().squeeze()

                state, reward, terminal, truncated, _=  self.Env.step( np.argmax( q_values  ) )
                
                rewards[-1] += reward
                if terminal or truncated :
                    if self.mode == "eval" :
                        state, _ = self.Env.reset()
                    break
  
        return np.mean(rewards)
        
    def create_gif(self,max_episodes:int=1  ) :
        images = []

        self.Env = gym.make(id=self.env_name, render_mode="rgb_array")
        state, _ = self.Env.reset()

        for _ in  range(max_episodes) :
            while True :

                frame = self.Env.render()      # RGB image (numpy array)
                images.append(frame)


                #print(images[0].shape)
                
                
                with torch.inference_mode() :
                        q_values = self.model(state).detach().cpu().data.numpy().squeeze()

                state, _, terminated, truncated, _ = self.Env.step(np.argmax( q_values  ))

                if terminated or truncated:
                    state, _ = self.Env.reset()
                    break

        print("num frames:", len(images))
        FileToSave = f"./experiments/figures/{self.env_name[:-3].lower()}.gif"
       
        with imageio.get_writer(FileToSave, mode="I",fps=30,loop=0 ) as writer:
             for img in images:
                 writer.append_data(img)

    def q_value_stats(self, q_values):
        """
        q_values: tensor of shape [batch_size, num_actions]
        """

        top2 = torch.topk(q_values, k=2, dim=1).values
        gap = top2[:, 0] - top2[:, 1]
        return {
            "q_mean": q_values.mean().item(),
            "q_max": q_values.max().item(),
            "q_min": q_values.min().item(),
            "q_std": q_values.std().item(),
            "mean_gap": gap.mean().item(),
            "min_gap": gap.min().item(),
            "max_gap": gap.max().item(),
        }
    
