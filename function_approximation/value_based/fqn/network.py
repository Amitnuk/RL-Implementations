import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple 
from common.networks.mlp import MLP

class FCQNetwork(nn.Module) :
    """
    This NN approximates Q(s,a) to Q(s,a;µ), where µ is the function parameters(weights) 
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

    
