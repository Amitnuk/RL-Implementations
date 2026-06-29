import torch
import numpy as np
from agent import FittedAgent
import os

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
        for episode in range(1, nb_episodes + 1) :
            
            state, _ = Agent.Env.reset(seed=12)
            self.episode_reward.append(0.0)   
            self.episode_reward_eval.append(0.0)
            while True :
                    
                state, is_terminated, experiences=Agent.act(state)
                self.episode_reward[-1] += experiences[2]
                
                Agent.store(experience=experiences)
                if len(Agent.buffer) >= Agent.batch_size :
                    experiences = Agent.load()

                    for _ in range(Agent.epochs) :
                        Agent.update(experiences=experiences)
                    
                    Agent.clear()
                    
                  
                total_step += 1
                if is_terminated :
                    break

            
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
            debug_message +=f"\nnumber of total steps: {total_step}\n"
            

            if self.best_agent_score < evaluation_score  or self.best_agent_score <= evaluation_score and evaluation_score == max_reward :
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
                CHECKPOINT_FILE = f"results/fqn/lunarlander/lunarlander_episode_nb_{episode}.pth"
                CHECKPOINT_DIR = os.path.join("./", CHECKPOINT_FILE)
                torch.save(checkpoint,CHECKPOINT_DIR)
                print("SAVING MODEL")


        
            print(debug_message)
            
