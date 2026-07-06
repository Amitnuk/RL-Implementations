import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple 
from common.networks.mlp import MLP

class FCQNetwork(nn.Module) :
    """
    Fully Connected Q Network
    This model approximates Q(s,a) to Q(s,a;µ), where µ is the function parameters(weights) 
    """
    def __init__(self,
                 input_shape:int,
                 output_shape:int,
                 hidden_units:Tuple=(256,256),
                 device:str="cpu") -> None:
        
        super(FCQNetwork, self).__init__()
        
        self.q_net = MLP(input_shape=input_shape,
                         output_shape=output_shape,
                         hidden_units=hidden_units,
                         activation_fn=F.relu)
        
        if torch.cuda.is_available() and device != "cpu" :
            device = "cuda"

        self.device = torch.device(device=device)
        self.to(device=self.device)
        print("FCQNetwork")
        
    def forward(self,state):
        """
        Given input state/observation, the network out puts Q

        Args
           state : the representation of what the agents sees.
        Output
           Q : action value function
        """
        if not isinstance(state,torch.Tensor) :
            state = torch.tensor(state,dtype=torch.float32, device=self.device)
            
        return self.q_net(state)


    def load(self,batch) :
        states, actions, rewards, next_states, is_terminals = batch

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, rewards, next_states, is_terminals
    
