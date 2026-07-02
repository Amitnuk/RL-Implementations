import torch
from torch.optim import Adam, Optimizer, RMSprop, SGD
import argparse
import gymnasium as gym
from trainer import Trainer
from agent import FittedAgent
from network import FCQNetwork
from common.policies.epsilon_greedy import EpsilonGreedyStrategy
from common.policies.softmax import SoftMaxStrategy


ENVIRONMENTS = {"cartpole":"CartPole-v1",
                "lunarlander":"LunarLander-v3",
                "mountaincar":"MountainCar-v0",
                "acrobot":"Acrobot-v1"}

POLICY_CHOICES = ["egreedy", "softmax", "gaussian", "greedy"]

def launch(args,Agent:FittedAgent) :
    trainer = Trainer()
    trainer.train(Agent=Agent,nb_episodes=args.max_episodes)
    
def select_policies(args) :

    if args.policy == "egreedy" :
        return EpsilonGreedyStrategy(epsilon=args.epsilon)
    elif args.policy == "softmax" :
        return SoftMaxStrategy()

    
def select_optimizer(args) :
    if args.optimizer == 0 :
        return lambda model, lr : Adam(params=model.parameters(),lr=lr)
    elif args.optimizer == 1 :
        return lambda model, lr : RMSprop(params=model.parameters(),lr=lr)
    else :
        return lambda model, lr : SGD(params=model.parameters(),lr=lr)
    
def parse_args() :
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--use_gpu",action="store_true",help="GPU usage, by the fault it is activated")
    parser.add_argument("--env",type=str,default="lunarlander", help="environment name,LunarLander is the the default environment")
    parser.add_argument("--optimizer",default=0,type=int,choices=[0,1,2],help="The optimizer to select, 0 for Adam, 1 for RMSProp and 2 for SGD, Adam is default")
    parser.add_argument("--max_episodes",default=1,type=int,help="Max episodes, default value is one")
    parser.add_argument("--buffer_size",default=10000,type=int,help="replay buffer size")
    parser.add_argument("--policy",default="egreedy",type=str,choices=POLICY_CHOICES,help="policy choices, egreedy is the default ")
    parser.add_argument("--epsilon",default=0.1,type=float, help="epsilon value for exploration")
    parser.add_argument("--gamma",default=0.99,type=float, help="discount value, default is 0.99")
    parser.add_argument("--batch_size",default=100,type=int, help="the size of the batch, default size is 100")
    parser.add_argument("--epochs",default=40,type=int, help="number of epochs, default is 40")
    parser.add_argument("--lr",default=0.003,type=float, help="learning rate, default is 0.003")

    
    return parser.parse_args()
    
def main() :
    args = parse_args()

    if args.env not in ENVIRONMENTS.keys() :
        print(f"Unknown environament named {args.env}")
        exit()
    if args.use_gpu :
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else :
        device = "cpu"
        
    
    print(f"{gym.pprint_registry()}")
    print(f"Training the agent in {args.env} environment on {args.use_gpu}")
    
    
    model = lambda nS, nA : FCQNetwork(input_shape=nS, output_shape=nA,hidden_units=(32,32),device=device)
    #buffer_size = args.batch_size if args.buffer_size > args.batch_size else args.buffer_size
    Agent = FittedAgent(env_name=ENVIRONMENTS[args.env],
                        value_model_fn=model,
                        value_optimizer_fn=select_optimizer(args),
                        value_optimizer_lr=args.lr,
                        training_strategy_fn=select_policies(args=args),
                        gamma=args.gamma,
                        batch_size=args.batch_size,
                        epochs=args.epochs)
    launch(args=args,
           Agent=Agent)
    
if __name__ == "__main__" :
    main()


