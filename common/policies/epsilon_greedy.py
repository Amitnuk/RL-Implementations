import torch
from torch import nn
import numpy as np



class GreedyStrategy :
    def __init__(self) :
        pass
        
    def select_discret_action(self,
                              model:nn.Module,
                              state:torch.Tensor) :

        with torch.inference_mode() :
            q_values = model(state).detach().cpu().data.numpy().squeeze()
        
        action = np.argmax(q_values)
            
        assert len(q_values) == q_values.shape[0]
        if not isinstance(action ,int) :
            action = action.item()

        return action
    

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
            
        assert len(q_values) == q_values.shape[0]
        if not isinstance(action ,int) :
            action = action.item()

        return action
    



class EGreedyLinearDecayStrategy :
    def __init__(self, max_steps, initial_epsilon:float=1, final_epsilon:float=0.1, decay_ratio=0.5) :
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.steps = 0
        self.decay_steps = max_steps * decay_ratio
        self.epsilon = initial_epsilon

    def decay_epsilon(self) :
        self.epsilon = 1 - self.steps/self.decay_steps
        self.epsilon = (self.initial_epsilon - self.final_epsilon)*self.epsilon + self.final_epsilon
        self.epsilon = np.clip(self.epsilon, self.final_epsilon, self.initial_epsilon)

        self.steps += 1
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

        self.decay_epsilon()
        return action
    



class EGreedyExponentialDecayStrategy :
    def __init__(self, max_steps, initial_epsilon:float=1, final_epsilon:float=0.1, decay_ratio:int=1) :
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = int(max_steps * decay_ratio)
        self.epsilons = 0.01/np.logspace(start=-2,stop=0, num=self.decay_steps,endpoint=False) - 0.01
        self.steps = 0
        self.epsilon = self.epsilons[self.steps]
        

    def decay_epsilon(self) :
        self.epsilon =  self.final_epsilon if self.steps >= self.decay_steps else self.epsilons[self.steps]
        self.epsilon = np.clip(self.epsilon, self.final_epsilon, self.initial_epsilon)
        self.steps += 1
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

        self.decay_epsilon()
        return action

