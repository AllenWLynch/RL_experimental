import gym
from utils import MarkovStateManager, RandomAgent, SequentialStateManager
import numpy as np

env = gym.make('CartPole-v1')
NUM_ACTIONS = 2

#manager = MarkovStateManager(RandomAgent(NUM_ACTIONS), env)
manager = SequentialStateManager(RandomAgent(2), env, lambda : np.zeros(4), seq_len = 10, overlap = 8)

for new_memory in manager.run():
    print(new_memory)
    input()
