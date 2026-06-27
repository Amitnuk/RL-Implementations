import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple 


class QNetwork(nn.Module) :
    """
    This NN approximates Q(s,a) to Q(s,a;µ), where µ is the function parameters(weights) 
    """
    def __init__(self,
                 input_shape:int,
                 output_shape:int,
                 hidden_units:Tuple=(256,256),
                 activation_fn=F.relu,
                 device:str="cpu") -> None:
        
        super(QNetwork, self).__init__()

        self.input_layer  = nn.Linear(in_features=input_shape, out_features=hidden_units[0])
        self.output_layer = nn.Linear(in_features=hidden_units[-1], out_features=output_shape)
        self.hidden_layers = nn.ModuleList()

        self.activation_fn = activation_fn
        for i in range(len(hidden_units)-1) :
            self.hidden_layers.append(nn.Linear(in_features= hidden_units[i],out_features=hidden_units[i+1]) )

        
        if torch.cuda.is_available() :
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
        
        if not isinstance(state, torch.Tensor) :
            state = torch.tensor(state,dtype=torch.float32,device=self.device)

        state = self.activation_fn(self.input_layer(state))

        for hidden_layer in self.hidden_layers :
            state = self.activation_fn(hidden_layer(state))

        return self.output_layer(state)

    
