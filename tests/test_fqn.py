import torch
from torch.optim import Adam, RMSprop, SGD
import numpy as np
from torch.nn import MSELoss
from function_approximation.value_based.fqn.network import FCQNetwork
from function_approximation.value_based.fqn.agent import FittedAgent

from common.policies.epsilon_greedy import EpsilonGreedyStrategy
from common.policies.softmax import SoftMaxStrategy
import gymnasium as gym
import argparse

if __name__ == "__main__" :

    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Tests")

    parser.add_argument("-e","--exp_strategy",default=0,type=int,choices=[0,1],help="Exploration strategy, Softmax -> 0 is default")
    parser.add_argument("--use_gpu",action="store_true",help="Use GPU")
    parser.add_argument("--optimizer",default=0,type=int,choices=[0,1,2],help="The optimizer to select, 0 for Adam, 1 for RMSProp and 2 for SGD, Adam is default")
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


    print(f"{gym.pprint_registry()}")

    model = lambda nS, nA : FCQNetwork(input_shape=nS, output_shape=nA,hidden_units=(256,256),device=device)

 
    optimizer = lambda model, lr :Adam(params=model.parameters(), lr=lr) if args.optimizer == 0 else RMSprop(params=model.parameters(), lr=lr) if args.optimizer == 1 else optim.optim.SGD(params=model.parameters(), lr=lr)

    env_name ="MountainCarContinuous-v0"

    Agent = FittedAgent(env_name=env_name,
                        value_model_fn=model,
                        value_optimizer_fn=optimizer,
                        value_optimizer_lr=0.01,
                        training_strategy_fn=Strategy
                        )
    Agent.interact()
