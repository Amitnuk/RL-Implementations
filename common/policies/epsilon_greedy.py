import torch
from torch import nn
import numpy as np

#eps = max(0.1, eps0 * decay_rate ** episode)
class EpsilonGreedyStrategy :
    def __init__(self, epsilon:float=0.1, final_epsilon:float=0.1) :
        self.epsilon = epsilon
        self.decay_ratio = 0.7
        self.initial_epsilon=1.0
        self.final_epsilon=final_epsilon
        self.decay_episodes = None     
        
    def set_initial_epsilon(self,initial_epsilon:float=1.0) :
        self.initial_epsilon = initial_epsilon

    def set_final_epsilon(self,final_epsilon:float=0.5) :
        self.final_epsilon = final_epsilon

    def get_epsilon(self) :
        return self.epsilon


    def set_decay_episodes_and_decay_ratio(self, nb_episodes:int, decay_ratio:int) :
        self.decay_ratio = decay_ratio
        self.decay_episodes = nb_episodes * self.decay_ratio

    def set_epsilon(self, epsilon) :
        self.epsilon = epsilon
        

    def decay(self, steps:int) :
        
        self.epsilon = 1 - steps/self.decay_episodes        
        self.epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon ) * self.epsilon                                                                                                                                               
        self.epsilon = np.clip( self.epsilon, self.final_epsilon, self.initial_epsilon)
        return self.epsilon

    def select_discret_action(self,
                              model:nn.Module,
                              state:torch.Tensor) :

        with torch.inference_mode() :
            q_values = model(state).detach().cpu().data.numpy().squeeze()
        
        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else :
            action = np.random.randint(len(q_values))
            
        assert len(q_values) == q_values.shape[0]
        if not isinstance(action ,int) :
            action = action.item()

        return action
    

