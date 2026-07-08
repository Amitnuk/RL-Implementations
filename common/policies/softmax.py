import torch
from torch import nn
import numpy as np


class SoftMaxStrategy :

    def __init__(self, init_temperature:float=1.0, min_temperature:float=0.3, exploration_ratio=0.8,  max_steps=25000) :

        self.init_temperature= init_temperature 
        self.min_temperature= min_temperature
        self.exploration_ratio= exploration_ratio
        self.max_steps= max_steps
        self.min = 0.1
        self.scale = 0.9999
        self.step = 0
        

    def decay_temperature(self) :

        temp = 1 -self.step/(self.exploration_ratio*self.max_steps)
        temp = (self.init_temperature - self.min_temperature)*temp + self.min_temperature 
        temp = np.clip(temp, self.min_temperature, self.init_temperature)
        self.step += 1

        return temp

    def select_discret_action(self, model:nn.Module, state:torch.Tensor) :
        
        
        self.temperature = self.decay_temperature()

        with torch.inference_mode() :
            q_values = model(state).detach().cpu().data.numpy().squeeze()

            scaled_q_values = q_values/self.temperature 
            norm_q_values = q_values - q_values.max()
            e = np.exp(q_values)
            probs = e/(np.sum(e)+ 1e-12)
            assert np.isclose(probs.sum(), 1.0)

        
        return np.random.choice(q_values.shape[0], p=probs)
        
        


