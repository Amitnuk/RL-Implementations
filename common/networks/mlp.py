import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple 


class MLP(nn.Module) :
    """
    A simple Multi Layer Perceptron 
    """
    def __init__(self,
                 input_shape:int,
                 output_shape:int,
                 hidden_units:Tuple,
                 activation_fn=F.relu) -> None:
    

        super(MLP, self).__init__()

        self.input_layer  = nn.Linear(in_features=input_shape,
                                      out_features=hidden_units[0])
        
        self.output_layer = nn.Linear(in_features=hidden_units[-1],
                                      out_features=output_shape)
        
        self.hidden_layers = nn.ModuleList()

        self.activation_fn = activation_fn
        for i in range(len(hidden_units)-1) :
            self.hidden_layers.append(nn.Linear(in_features= hidden_units[i],
                                                out_features=hidden_units[i+1]) )

        
        
    def forward(self,x):
        
        if not isinstance(x, torch.Tensor) :
            x = torch.tensor(x,dtype=torch.float32,device=self.device)

        x = self.activation_fn(self.input_layer(x))

        for hidden_layer in self.hidden_layers :
            x = self.activation_fn(hidden_layer(x))

        return self.output_layer(x)
