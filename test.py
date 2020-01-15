import gym
from utils import MarkovStateManager, RandomAgent, SequentialStateManager, PrioritizedLearner, PolicyAgent, train_prioritized_undistributed
import numpy as np
from SAC_algos import MarkovDuelingDiscreteSAC
from prioritized_exp.prioritized_memory import Memory
import networks
import tensorflow as tf

env = gym.make('CartPole-v1')
NUM_ACTIONS = 2

#manager = SequentialStateManager(RandomAgent(2), env, lambda : np.zeros(4), seq_len = 10, overlap = 8)

q_net = networks.SimpleDueling((50,50), (4,),NUM_ACTIONS)

policy_net = networks.SimpleDense((50,50))
policy_net.add(tf.keras.layers.Dense(NUM_ACTIONS))
policy_net.add(tf.keras.layers.Softmax())

actor = MarkovStateManager(PolicyAgent(policy_net), env, reward_fn = lambda x : max(-1.0, min(1.0, 0.05 * x)))
state_shape = actor.get_state_shape()

learner = PrioritizedLearner(MarkovDuelingDiscreteSAC(policy_net, q_net, state_shape, NUM_ACTIONS), 1000, 64)

replay = Memory(int(1e6))

train_prioritized_undistributed(actor, learner, replay)





'''for new_memory in manager.run():
    print(new_memory)
    input()'''
