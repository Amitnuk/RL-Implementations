import torch
from torch.optim import Adam, Optimizer, RMSprop, SGD
import argparse
import gymnasium as gym
from trainer import Trainer
from agent import NFQIAgent
from network import FCQNetwork
from common.policies.epsilon_greedy import EpsilonGreedyStrategy
from common.policies.softmax import SoftMaxStrategy
import os

ENVIRONMENTS = {"cartpole":"CartPole-v1",
                "lunarlander":"LunarLander-v3",
                "mountaincar":"MountainCar-v0",
                "acrobot":"Acrobot-v1"}

BEHAVIOUR_POLICY_CHOICES = ["egreedy", "softmax", "gaussian", "greedy"]
MODES = ["eval", "train"]



def run_launcher(args,Agent:NFQIAgent,lowest_evaluation_score:int) :
    trainer = Trainer(lowest_evaluation_score=lowest_evaluation_score)
    trainer.train(Agent=Agent,nb_episodes=args.max_episodes, ENV=args.env,DECAY_RATIO=args.decay_ratio)
    
def select_policies(args) :

    if args.behaviour_policy == "egreedy" :
        return EpsilonGreedyStrategy(epsilon=args.epsilon, final_epsilon=args.final_epsilon)
    elif args.behaviour_policy == "softmax" :
        return SoftMaxStrategy()
    else :
        print(f"{args.behaviour_policy} is not yet implemented, so epsilon greedy will be selected instead")
        return EpsilonGreedyStrategy(epsilon=args.epsilon)

    
def select_optimizer(args) :
    if args.optimizer == 0 :
        return lambda model, lr : Adam(params=model.parameters(),lr=lr)
    elif args.optimizer == 1 :
        return lambda model, lr : RMSprop(params=model.parameters(),lr=lr)
    else :
        return lambda model, lr : SGD(params=model.parameters(),lr=lr)
    
def parse_args() :
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--launch", type=str, choices=MODES, default=MODES[0], help="running mode, train or test, test is the default")
    parser.add_argument("--use_gpu",action="store_true",help="GPU usage, by the fault it is activated")
    parser.add_argument("--env",type=str,default="lunarlander", help="environment name,LunarLander is the the default environment")
    parser.add_argument("--max_episodes",default=1,type=int,help="Max episodes, default value is one")
    parser.add_argument("--buffer_size",default=10000,type=int,help="replay buffer size")
    parser.add_argument("--gamma",default=0.99,type=float, help="discount value, default is 0.99")
    parser.add_argument("--behaviour_policy",default="egreedy",type=str,choices=BEHAVIOUR_POLICY_CHOICES,help="policy choices, egreedy is the default ")
    parser.add_argument("--batch_size",default=100,type=int, help="the size of the batch, default size is 100")
    parser.add_argument("--epochs",default=40,type=int, help="number of epochs, default is 40")
    parser.add_argument("--optimizer",default=0,type=int,choices=[0,1,2],help="The optimizer to select, 0 for Adam, 1 for RMSProp and 2 for SGD, Adam is default")
    parser.add_argument("--lr",default=0.003,type=float, help="learning rate, default is 0.003")
    parser.add_argument("--epsilon",default=0.1,type=float, help="epsilon value for exploration")
    parser.add_argument("--final_epsilon", default=0.1, type=float, help="the lower epsilon can get for when epsilon when epsilon greedy is selected, default is one")
    parser.add_argument("--decay_ratio", default=1.0, type=float, help="decay ratio for when epsilon when epsilon greedy is selected, default is one")
    parser.add_argument("--seed", default=34,type=int, help="seeding, the default value is 34")

    
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
    print(f"Training the agent in {args.env} environment on Cuda : {args.use_gpu}")
    
    
    if args.env == "cartpole" :
        hidden_units = (512,128)
        lowest_evaluation_score = 0
    elif args.env == "lunarlander":
        hidden_units = (256,256)
        lowest_evaluation_score = 0
    elif args.env == "acrobot" :
        hidden_units = (256,256) 
        lowest_evaluation_score = -100
    else :
        hidden_units = (4,4) 
        lowest_evaluation_score = -200

    if args.launch != MODES[1] and args.launch != MODES[0] :
        print("must choose the launcing mode, train or eval")
        exit()

    model = lambda nS, nA : FCQNetwork(input_shape=nS, output_shape=nA,hidden_units=hidden_units,device=device)
    
    Agent = NFQIAgent(env_name=ENVIRONMENTS[args.env],
                        value_model_fn=model,
                        value_optimizer_fn=select_optimizer(args),
                        value_optimizer_lr=args.lr,
                        training_strategy_fn=select_policies(args=args),
                        gamma=args.gamma,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        seed=args.seed,
                        mode=args.launch)
    

    if args.launch == MODES[1] : 
        print("Training Mode")
        run_launcher(args=args,Agent=Agent, lowest_evaluation_score=lowest_evaluation_score)

    if args.launch == MODES[0] :
        CHECKPOINT_FILE = f"./results/nfqi/{args.env}/{args.env}_best.pth"
        if os.path.isfile(CHECKPOINT_FILE) :
            checkpoint = torch.load(CHECKPOINT_FILE,weights_only=False)
            Agent.model.load_state_dict(checkpoint['model_state_dict'])
            print(checkpoint["best_score"])
        else :
            print("dod not found the model path")
            exit()
        print("Evaluation Mode")
        
        
        Agent.evaluate(args.max_episodes)
        #Agent.create_gif()

        
    
if __name__ == "__main__" :
    main()


