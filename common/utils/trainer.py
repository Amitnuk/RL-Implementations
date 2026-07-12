import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from itertools import count


class Trainer :
    def __init__(self, lowest_evaluation_score = -200) :
        self.lowest_evaluation_score = lowest_evaluation_score


    def fqi_debug_step(self,Agent, states):
        with torch.inference_mode():
            q = Agent.online_model(states)

        q_stats = Agent.q_value_stats(q)

        return {**q_stats}
    
    def train(self,
              Agent,
              nb_episodes:int,
              ENV:str) :

 
        self.best_agent_score = -1000 
        total_step = 0
        evaluation_score = 0


        AgentName = Agent.__class__.__name__
        if "Agent" not in AgentName :
            print(f"The Agent Name is Wrong, it shoud be AlgorithmAgent, eg:(DQNAgent or SACAgent), instead if {AgentName}")
            exit()

        algorithm = AgentName.replace("Agent", "").lower()
        TENSORBOARD_PATH = f"runs/{algorithm}/{Agent.env_name[:-3].lower()}"
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
                         "max_gap": 0,
                         "loss":0}
        
        total_step = 0
        best_episode = -1
        for episode in range(1, nb_episodes + 1) :
            print("TRAIN")
            debug_results = Agent.interact(episode, egreedy_like_p, self.writer, self.fqi_debug_step, debug_results)
            total_step += 1

            print("EVAL")
            EVAL_MODE = True
            evaluation_score, done, evaluation_std = Agent.evaluate()
            Agent.episode_reward_eval[-1] = evaluation_score
            Agent.episode_reward_std_eval[-1] = evaluation_std
            
            debug_message =f"\nepisode: {episode}"
            debug_message +=f"\ntraining score last episode: {Agent.episode_reward[-1]:.2f}"
            debug_message +=f"\ntraining score for last 10 episodes : {np.mean(Agent.episode_reward[-10:]):.2f}"
            debug_message +=f"\ntraining score for last 100 episodes : {np.mean(Agent.episode_reward[-100:]):.2f}"
            debug_message +=f"\ntraining score for last 1000 episodes : {np.mean(Agent.episode_reward[-1000:]):.2f}"
            debug_message +=f"\ntraining score for all episodes: {np.mean(Agent.episode_reward):.2f}"
            debug_message +=f"\ntraining episode_timestep : {np.sum(Agent.episode_timestep):.2f}"
            debug_message +=f"\nnumber of total steps: {total_step}"

            if EVAL_MODE :

                debug_message +=f"\ntraining loss : {debug_results['loss']:.5f}"
                debug_message +=f"\nq_mean : {debug_results['q_mean']}"
                debug_message +=f"\nq_max : {debug_results['q_max']}"
                debug_message +=f"\nq_min: {debug_results['q_min']}"
                debug_message +=f"\nq_std : {debug_results['q_std']}"
                debug_message +=f"\nmean_gap : {debug_results['mean_gap']}"
                debug_message +=f"\nmin_gap : {debug_results['min_gap']}"
                debug_message +=f"\nmax_gap: {debug_results['max_gap']}"

            debug_message +=f"\nevaluation score last episode: {Agent.episode_reward_eval[-1]:.2f}"
            debug_message +=f"\nevaluation score for last 10 episodes : {np.mean(Agent.episode_reward_eval[-10:]):.2f}"
            debug_message +=f"\nevaluation std score for last 10 episodes : {np.mean(Agent.episode_reward_std_eval[-10:]):.2f}"

            debug_message +=f"\nevaluation score for last 100 episodes : {np.mean(Agent.episode_reward_eval[-100:]):.2f}"
            debug_message +=f"\nevaluation std score for last 100 episodes : {np.mean(Agent.episode_reward_std_eval[-100:]):.2f}"

            debug_message +=f"\nevaluation score for last 1000 episodes : {np.mean(Agent.episode_reward_eval[-1000:]):.2f}"
            debug_message +=f"\nevaluation std score for last 1000 episodes : {np.mean(Agent.episode_reward_std_eval[-1000:]):.2f}"

            debug_message +=f"\nevaluation score for all episodes : {np.mean(Agent.episode_reward_eval):.2f}"
            debug_message +=f"\nevaluation std score for all episodes : {np.mean(Agent.episode_reward_std_eval):.2f}"


            if egreedy_like_p :
                debug_message +=f"\nepsilon : {behavior_policy.epsilon:.4f}"
            
            
            self.writer.add_scalar("Train/Reward",Agent.episode_reward[-1], episode)
            self.writer.add_scalar("Train/RewardMean",np.mean(Agent.episode_reward), episode)
            self.writer.add_scalar("Train/AverageRewardMean10",np.mean(Agent.episode_reward[-10:]),episode)
            self.writer.add_scalar("Train/AverageRewardMean100",np.mean(Agent.episode_reward[-100:]),episode)
            self.writer.add_scalar("Train/AverageRewardMean1000",np.mean(Agent.episode_reward[-1000:]),episode)

            self.writer.add_scalar("Eval/Reward",Agent.episode_reward_eval[-1], episode)
            self.writer.add_scalar("Eval/RewardMean",np.mean(Agent.episode_reward_eval), episode)
            self.writer.add_scalar("Eval/AverageRewardMean10",np.mean(Agent.episode_reward_eval[-10:]),episode)
            self.writer.add_scalar("Eval/AverageRewardMean100",np.mean(Agent.episode_reward_eval[-100:]),episode)
            self.writer.add_scalar("Eval/AverageRewardMean1000",np.mean(Agent.episode_reward_eval[-1000:]),episode)

            self.writer.add_scalar("Eval/StdRewardMean",np.mean(Agent.episode_reward_std_eval), episode)
            self.writer.add_scalar("Eval/AverageStdRewardMean10",np.mean(Agent.episode_reward_std_eval[-10:]),episode)
            self.writer.add_scalar("Eval/AverageStdRewardMean100",np.mean(Agent.episode_reward_std_eval[-100:]),episode)
            self.writer.add_scalar("Eval/AverageStdRewardMean1000",np.mean(Agent.episode_reward_std_eval[-1000:]),episode)

            if self.best_agent_score <= evaluation_score  and evaluation_score > self.lowest_evaluation_score :
                checkpoint = {
                                "model":Agent.online_model,
                                "model_state_dict":Agent.online_model.state_dict(),
                                "optimizer":Agent.optimizer,
                                "optimizer_state_dict":Agent.optimizer.state_dict(),
                                "learning_rate":Agent.learning_rate,
                                "epochs":Agent.epochs,
                                "env_name":Agent.env_name,
                                "best_score":evaluation_score}
                
                self.best_agent_score = evaluation_score
                best_episode = episode
                CHECKPOINT_FILE = f"results/{algorithm}/{ENV}/{ENV}_best.pth"
                CHECKPOINT_DIR = os.path.join("./", CHECKPOINT_FILE)
                torch.save(checkpoint,CHECKPOINT_DIR)
                print("SAVING MODEL")
                
            debug_message+=f"\nbest evaluation score so far: {self.best_agent_score} at episode: {best_episode}"

            os.system("clear")
            print(debug_message)
            
        self.writer.close()
