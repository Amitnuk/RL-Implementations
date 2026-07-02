import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from agent import FittedAgent
import os
from itertools import count
class Trainer :
    def __init__(self) :
        pass

    def train(self,
              Agent:FittedAgent,
              nb_episodes:int,
              max_reward:int=200) :

        self.episode_reward = [] 
        self.episode_reward_eval = []
        self.best_agent_score = -1000 
        total_step = 0
        evaluation_score = 0
        evaluation_best_score = 0
        
        LOG_PATH = "common/loggins/mountaincar/0/"
       
        if not os.path.exists(LOG_PATH) :
            print(f"{LOG_PATH} does not exits ")
            exit()

        self.writer = SummaryWriter(LOG_PATH)
        for episode in range(1, nb_episodes + 1) :
            
            state, _ = Agent.Env.reset()
            self.episode_reward.append(0.0)   
            self.episode_reward_eval.append(0.0)
            print("TRAIN")
            for step in count() :
                    
                state, is_terminated, experiences=Agent.act(state)
               
                self.episode_reward[-1] += experiences[2]
                #print("state=",experiences[0],'\naction=',experiences[1],'\nreward=',experiences[2],'\nnext_state=',experiences[3],'\nterminal=',experiences[4],'\n')
                Agent.store(experience=experiences)
                if len(Agent.buffer) >= Agent.batch_size :
                    batch = Agent.buffer.load(Agent.batch_size)
                    experiences = Agent.model.load(batch)
                    print("UPDATE")
                    for _ in range(Agent.epochs) :
                        Agent.update(experiences=experiences)

                    Agent.clear()
                    evaluate = True
                    

                total_step += 1
                if is_terminated :
                    break

            
            print("EVALUATE")
            evaluation_score = Agent.evaluate()
            self.episode_reward_eval[-1] = evaluation_score

            
            debug_message =f"\nepisode: {episode}"
            debug_message +=f"\ntraining score: {np.mean(self.episode_reward):.2f}"
            debug_message +=f"\ntraining score last episodes: {self.episode_reward[-1]:.2f}"
            debug_message +=f"\ntraining score for last 10 episodes : {np.mean(self.episode_reward[-10:]):.2f}"
            debug_message +=f"\ntraining score for last 100 episodes : {np.mean(self.episode_reward[-100:]):.2f}"
            debug_message +=f"\nevaluation score: {evaluation_score:.2f}"
            debug_message +=f"\nevaluation score for last 10 episodes : {np.mean(self.episode_reward_eval[-10:]):.2f}"
            debug_message +=f"\nevaluation score for last 100 episodes : {np.mean(self.episode_reward_eval[-100:]):.2f}"
            debug_message +=f"\nnumber of total steps: {total_step}"
            
            self.writer.add_scalar("Train/Reward",np.mean(self.episode_reward), episode)
            self.writer.add_scalar("Train/AverageReward10",np.mean(self.episode_reward[-10:]),episode)
            self.writer.add_scalar("Train/AverageReward100",np.mean(self.episode_reward[-100:]),episode)
            self.writer.add_scalar("Train/AverageReward1000",np.mean(self.episode_reward[-1000:]),episode)

            self.writer.add_scalar("Eval/Reward",np.mean(self.episode_reward_eval), episode)
            self.writer.add_scalar("Eval/AverageReward10",np.mean(self.episode_reward_eval[-10:]),episode)
            self.writer.add_scalar("Eval/AverageReward100",np.mean(self.episode_reward_eval[-100:]),episode)
            self.writer.add_scalar("Eval/AverageReward1000",np.mean(self.episode_reward_eval[-1000:]),episode)

            if self.best_agent_score <= evaluation_score and evaluation_score > -200 :
                checkpoint = {
                                "model":Agent.model,
                                "model_state_dict":Agent.model.state_dict(),
                                "optimizer":Agent.optimizer,
                                "optimizer_state_dict":Agent.optimizer.state_dict(),
                                "learning_rate":Agent.learning_rate,
                                "epochs":Agent.epochs,
                                "env_name":Agent.env_name,
                                "best_score":evaluation_score}
                
                self.best_agent_score = evaluation_score
                evaluation_best_score = evaluation_score
                CHECKPOINT_FILE = f"results/fqn/mountaincar/mountaincar_episode_nb_{episode}.pth"
                CHECKPOINT_DIR = os.path.join("./", CHECKPOINT_FILE)
                torch.save(checkpoint,CHECKPOINT_DIR)
                print("SAVING MODEL")
                
            debug_message+=f"\nbest eveluation score so far: {evaluation_best_score}"

            os.system("clear")
            print(debug_message)
            
        self.writer.close()
