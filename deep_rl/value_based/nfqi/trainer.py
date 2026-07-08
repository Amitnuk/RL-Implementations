import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from agent import NFQIAgent
import os
from itertools import count


class Trainer :
    def __init__(self, lowest_evaluation_score = -200) :
        self.lowest_evaluation_score = lowest_evaluation_score


    def fqi_debug_step(self,Agent:NFQIAgent, states):
        with torch.inference_mode():
            q = Agent.model(states)

        q_stats = Agent.q_value_stats(q)

        return {**q_stats}
    
    def train(self,
              Agent:NFQIAgent,
              nb_episodes:int,
              ENV:str) :

        self.episode_reward = [] 
        self.episode_reward_eval = []
        self.best_agent_score = -1000 
        total_step = 0
        evaluation_score = 0
        evaluation_best_score = self.lowest_evaluation_score
        
        TENSORBOARD_PATH = f"runs/nfqi/{Agent.env_name[:-3].lower()}"
        behavior_policy = Agent.training_strategy_fn
        egreedy_like_p = False

        if hasattr(behavior_policy, "epsilon") :
            print(behavior_policy.epsilon)
            egreedy_like_p = True


        if not os.path.exists(TENSORBOARD_PATH) :
            print(f"{TENSORBOARD_PATH} does not exits ")
            

        self.writer = SummaryWriter(TENSORBOARD_PATH)
       
        EVAL_MODE = False

        debug_results = {"q_mean": 0,
                         "q_max": 0,
                         "q_min": 0,
                         "q_std": 0,
                         "mean_gap": 0,
                         "min_gap": 0,
                         "max_gap": 0}
        

        train_step = 0
        for episode in range(1, nb_episodes + 1) :
            
            
            self.episode_reward.append(0.0)   
            self.episode_reward_eval.append(0.0)
            state, _ = Agent.Env.reset(seed=Agent.seed + episode -1)

            
            print("TRAIN")
            for step in count() :
                state, is_terminated, experiences=Agent.act(state)
                Agent.training_strategy_fn.decay_eps = False
                
                self.episode_reward[-1] += experiences[2]
                Agent.store(experience=experiences)
                if len(Agent.buffer) >= Agent.batch_size :
                    batch = Agent.buffer.load(Agent.batch_size)
                    experiences = Agent.model.load(batch)
                    
                    for _ in range(Agent.epochs) :
                        Agent.update(experiences=experiences)
                    loss = Agent.loss
                    
                    debug_results = self.fqi_debug_step(Agent, experiences[0])
                   
                    self.writer.add_scalar("Train/Loss",loss, train_step)
                    if egreedy_like_p :
                        self.writer.add_scalar("Train/Epsilon",behavior_policy.epsilon, episode)
                    train_step += 1
                    Agent.clear()
                    EVAL_MODE = True
                    

                total_step += 1
                if is_terminated :
                    break

            print("EVAL")
            evaluation_score = Agent.evaluate()
            self.episode_reward_eval[-1] = evaluation_score

            
            debug_message =f"\nepisode: {episode}"
            debug_message +=f"\ntraining score: {np.mean(self.episode_reward):.2f}"
            debug_message +=f"\ntraining score last episodes: {self.episode_reward[-1]:.2f}"
            debug_message +=f"\ntraining score for last 10 episodes : {np.mean(self.episode_reward[-10:]):.2f}"
            debug_message +=f"\ntraining score for last 100 episodes : {np.mean(self.episode_reward[-100:]):.2f}"
            debug_message +=f"\ntraining score for last 1000 episodes : {np.mean(self.episode_reward[-1000:]):.2f}"
            debug_message +=f"\nnumber of total steps: {total_step}"

            if EVAL_MODE :
                debug_message +=f"\ntraining loss : {loss:.5f}"
                debug_message +=f"\nq_mean : {debug_results['q_mean']}"
                debug_message +=f"\nq_max : {debug_results['q_max']}"
                debug_message +=f"\nq_min: {debug_results['q_min']}"
                debug_message +=f"\nq_std : {debug_results['q_std']}"
                debug_message +=f"\nmean_gap : {debug_results['mean_gap']}"
                debug_message +=f"\nmin_gap : {debug_results['min_gap']}"
                debug_message +=f"\nmax_gap: {debug_results['max_gap']}"

            debug_message +=f"\nevaluation score: {evaluation_score:.2f}"
            debug_message +=f"\nevaluation score for last 10 episodes : {np.mean(self.episode_reward_eval[-10:]):.2f}"
            debug_message +=f"\nevaluation score for last 100 episodes : {np.mean(self.episode_reward_eval[-100:]):.2f}"
            debug_message +=f"\nevaluation score for last 1000 episodes : {np.mean(self.episode_reward_eval[-100:]):.2f}"

            if egreedy_like_p :
                debug_message +=f"\nepsilon : {behavior_policy.epsilon:.4f}"
            
            
            self.writer.add_scalar("Train/Reward",np.mean(self.episode_reward), episode)
            self.writer.add_scalar("Train/AverageReward10",np.mean(self.episode_reward[-10:]),episode)
            self.writer.add_scalar("Train/AverageReward100",np.mean(self.episode_reward[-100:]),episode)
            self.writer.add_scalar("Train/AverageReward1000",np.mean(self.episode_reward[-1000:]),episode)

            self.writer.add_scalar("Eval/Reward",np.mean(self.episode_reward_eval), episode)
            self.writer.add_scalar("Eval/AverageReward10",np.mean(self.episode_reward_eval[-10:]),episode)
            self.writer.add_scalar("Eval/AverageReward100",np.mean(self.episode_reward_eval[-100:]),episode)
            self.writer.add_scalar("Eval/AverageReward1000",np.mean(self.episode_reward_eval[-1000:]),episode)

            if self.best_agent_score <= evaluation_score  and evaluation_score > self.lowest_evaluation_score and 0:
                
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
                CHECKPOINT_FILE = f"results/nfqi/{ENV}/{ENV}_episode_nb_{episode}.pt"
                CHECKPOINT_DIR = os.path.join("./", CHECKPOINT_FILE)
                torch.save(checkpoint,CHECKPOINT_DIR)
                print("SAVING MODEL")
                
            debug_message+=f"\nbest eveluation score so far: {evaluation_best_score}"

            os.system("clear")
            print(debug_message)
            
        self.writer.close()
