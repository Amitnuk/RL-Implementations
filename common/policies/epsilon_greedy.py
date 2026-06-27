import torch
from torch import nn
import numpy as np


class EpsilonGreedyStrategy() :
    def __init__(self, epsilon:float=0.1) :
        self.epsilon = epsilon

    def select_action(self,
                      model:nn.Module,
                      state:torch.Tensor) :

        with torch.inference_mode() :
            q_values = model(state).detach().cpu().numpy().squeeze()
        if np.random.randn() > self.epsilon:
            action = np.argmax(q_values)
        else :
            action = np.random.randint(q_values.shape[0])
        return action
