import torch
import numpy as np
from typing import Tuple

class ReplayBuffer :
    def __init__(self,
                 max_size:int=10000,
                 batch_size=512) :

        self.states      = np.empty(shape=(max_size),dtype=np.ndarray)
        self.actions     = np.empty(shape=(max_size),dtype=np.ndarray)
        self.next_states = np.empty(shape=(max_size),dtype=np.ndarray)
        self.rewards     = np.empty(shape=(max_size),dtype=np.ndarray)
        self.terminals   = np.empty(shape=(max_size),dtype=np.ndarray)

        self.max_size = max_size
        self.batch_size = batch_size
        self.idx = 0
        self.size = 0

    def clear(self) :

        self.states      = np.empty(shape=(self.max_size),dtype=np.ndarray)
        self.actions     = np.empty(shape=(self.max_size),dtype=np.ndarray)
        self.next_states = np.empty(shape=(self.max_size),dtype=np.ndarray)
        self.rewards     = np.empty(shape=(self.max_size),dtype=np.ndarray)
        self.terminals   = np.empty(shape=(self.max_size),dtype=np.ndarray)


        self.idx = 0
        self.size = 0
        
    def load(self) :
        
        experience = np.vstack(self.states[:self.size]),\
                     np.vstack(self.actions[:self.size]), \
                     np.vstack(self.rewards[:self.size]),\
                     np.vstack(self.next_states[:self.size]),\
                     np.vstack(self.terminals[:self.size])
        
        return experience
    def sample(self, batch_size=None) :


        if batch_size != None :
            self.batch_size = batch_size

        idx = np.random.choice(self.size,self.batch_size)

        experience = np.vstack(self.states[idx]),\
                     np.vstack(self.actions[idx]), \
                     np.vstack(self.rewards[idx]),\
                     np.vstack(self.next_states[idx]),\
                     np.vstack(self.terminals[idx])

        return experience


    def store(self, experience:Tuple ) ->None :
        state, action, reward, next_state,terminal = experience

        self.states[self.idx]      = state
        self.actions[self.idx]     = action
        self.rewards[self.idx]     = reward
        self.next_states[self.idx] = next_state
        self.terminals[self.idx]   = terminal

        self.idx +=1
        self.idx = self.idx % self.max_size
        self.size +=1
        self.size = min(self.size,self.max_size)
    def __len__(self) :
        return self.size 
