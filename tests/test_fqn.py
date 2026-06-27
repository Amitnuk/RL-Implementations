import torch
import numpy as np
from function_approximation.value_based.fqn.network import FCQNetwork
from common.policies.epsilon_greedy import EpsilonGreedyStrategy
from common.policies.softmax import SoftMaxStrategy
import argparse

if __name__ == "__main__" :

    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Tests")

    parser.add_argument("-e","--exp_strategy",default=0,type=int,choices=[0,1],help="Exploration strategy, Softmax -> 0 is default")
    parser.add_argument("--use_gpu",action="store_true",help="Use GPU")
    args = parser.parse_args()

    
    if args.use_gpu :
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else :
        device = "cpu"



    state = torch.randn(size=(5,),device=torch.device(device))
    
    QNet = FCQNetwork(input_shape=state.shape[0],
                      output_shape=3,
                      hidden_units=(5,5),
                      device=device)

    Q = QNet(state)

    Strategy = [EpsilonGreedyStrategy(), SoftMaxStrategy(temperature=0.5)][args.exp_strategy]
    
    print(f"Q={Q, Q.shape}")

    action = Strategy.select_action(model=QNet,state=state)
    print(f"action={action, type(action)}")
    
