import torch
from torch import nn
import numpy as np


class EpsilonGreedyStrategy :
    def __init__(self, epsilon:float=0.1) :
        self.epsilon = epsilon

    def select_discret_action(self,
                              model:nn.Module,
                              state:torch.Tensor) :

        with torch.inference_mode() :
            q_values = model(state).detach().cpu().data.numpy().squeeze()
        
        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else :
            action = np.random.randint(len(q_values))
            #print("action:",action)
            assert len(q_values) == q_values.shape[0]
        if not isinstance(action ,int) :
            action = action.item()
        return action
