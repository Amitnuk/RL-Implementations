import torch
from torch import nn
from torch import optim
from torch.nn import MSELoss
import numpy as np 
import gymnasium as gym 
from gymnasium.spaces import Box, Discrete
from common.buffers.replaybuffer import ReplayBuffer

class FittedQNet() :
    def __init__(self,
                 Env:gym.Env,
                 value_model_fn,
                 value_optimizer_fn,
                 value_optimizer_lr:float,
                 training_strategy_fn,
                 batch_size:int=32) :

        self.Env = Env
        torch.manual_seed(34)
        torch.cuda.manual_seed(34)
        #np.random.seed(12)
        

        #self.Env.observation_space.seed(12)
        # NQF works with discrete actions
        #if isinstance(self.Env.action_space,Box) :
        #    self.action_dim = self.Env.action_space.shape[0]
        #    self.low  = self.Env.action_space.low.item()
        #    self.high = self.Env.action_space.high.item() 
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

        
        self.online_model = value_model_fn(self.nS, self.nA)
        
        self.value_optimizer_lr   = value_optimizer_lr
        self.optimizer    = value_optimizer_fn(self.online_model,self.value_optimizer_lr)
        self.training_strategy_fn = training_strategy_fn
        self.batch_size=  batch_size

        print("FittedQNet")

    def iteration_step(self, state) :
        
        action = self.training_strategy_fn.select_discret_action(self.online_model,state)
        new_state, reward, terminal, truncated, _ = self.Env.step(action=action)
        is_terminated = terminal or truncated
        experience = (state, action, reward, new_state, float(terminal))
        return new_state, is_terminated, experience 
 
    
class FittedAgent :
    def __init__(self,
                 env_name:str,
                 value_model_fn,
                 value_optimizer_fn,
                 value_optimizer_lr:float,
                 training_strategy_fn,
                 gamma:float=0.99,
                 batch_size:int=100,
                 epochs:int=40) :
        
        self.env_name = env_name
        print(f"Selected env = {self.env_name}")        
        self.Env = gym.make(id=self.env_name,render_mode="human") 
        
        QNet = FittedQNet(Env=self.Env,
                          value_model_fn=value_model_fn,
                          value_optimizer_fn=value_optimizer_fn,
                          value_optimizer_lr=value_optimizer_lr,
                          training_strategy_fn=training_strategy_fn)

        self.gamma = gamma
        self.batch_size =batch_size
        self.model = QNet.online_model
        self.optimizer = QNet.optimizer
        self.iteration_step  = QNet.iteration_step
        self.epochs = epochs
        self.buffer = ReplayBuffer(batch_size=self.batch_size)
        self.learning_rate = value_optimizer_lr
        print("FittedAgent")

    def act(self,state) :
        return self.iteration_step(state=state)
        
    def load(self):
        return self.buffer.load()
    
    def store(self,experience) :
        self.buffer.store(experience=experience)
    def clear(self) :
         self.buffer.clear()
    def update(self, experiences) :
        states, actions, rewards, next_states, is_terminated = experiences
        
        #print(f"states={states}\naction={actions}\nrewards={rewards}\next_states={next_states}\nterminals={is_terminated}" )
        #exit()
        max_a_q_sp = self.model(next_states).detach().max(1)[0].unsqueeze(1)
        
        td_target_qs = rewards +  self.gamma * max_a_q_sp * ( 1 - is_terminated )
        
        q_sa     = self.model(states).gather(1, actions)
        
        #print(td_target_qs,"\n", q_sa)
        #exit()
        
        td_errors  = td_target_qs - q_sa

        value_losses = td_errors.pow(2).mul(0.5).mean()

        self.optimizer.zero_grad()
        value_losses.backward()
        self.optimizer.step()

    def evaluate(self):
        rewards=[]
        state, _ = self.Env.reset()
        
        for _ in range(1) :
            rewards.append(0.0)
            while True :
                
                with torch.inference_mode() :
                    q_values = self.model(state).detach().cpu().data.numpy().squeeze()

                state, reward, terminal, truncated, _=  self.Env.step( np.argmax( q_values  ) )
                
                rewards[-1] += reward
                if terminal or truncated :
                    break
  
        return np.mean(rewards)
        
    
