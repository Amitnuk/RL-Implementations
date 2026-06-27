import torch
from torch import nn
import numpy as np


class SoftMaxStrategy :

    def __init__(self, temperature:float=1.0) :
        self.temperature = temperature


    def select_action(self, model:nn.Module, state:torch.Tensor) :
        
        q_values = model(state).detach().cpu().numpy()

        
        q_values = q_values - q_values.max()
        exp_q = np.exp(q_values/self.temperature)
        probs = exp_q/exp_q.sum()
        return np.random.choice(q_values.shape[0], p=probs)
        
        


