import torch
from torch import nn
import numpy as np


class SoftMaxStrategy :

    def __init__(self, temperature:float=1.0) :
        self.base= temperature * 1.5
        self.min = 0.1
        self.scale = 0.9999
        self.step = 0
        self.warmup_steps = 1700000

    def select_discret_action(self, model:nn.Module, state:torch.Tensor) :
        
        self.step += 1

        if self.step < self.warmup_steps :
            self.temperature = self.base
        else :
            self.temperature = max(self.min, self.base*self.scale**(self.step  - self.warmup_steps))


        q_values = model(state).detach().cpu().data.numpy().squeeze()

        q_values = q_values - q_values.max()
        q_values = q_values/self.temperature 

        exp_q = np.exp(q_values)
        probs = exp_q/(exp_q.sum() + 1e-12)

        #probs = 0.8 * probs + 0.2 / q_values.shape[0]
        probs = probs/sum(probs)
        #if self.step % 50 == 0 :
        #    print(f"action selected: {np.random.choice(q_values.shape[0], p=probs)}\nprobs: {probs}\nq: {q_values}")


        return np.random.choice(q_values.shape[0], p=probs)
        
        


