
import tensorflow as tf
import gym
from utils import MarkovStateManager, NaiveReplay, PrioritizedLearner
import numpy as np
from SAC_algos import MarkovDuelingDiscreteSAC
import networks
import tensorflow as tf
from actors import PolicyAgent, BasicActor
from prioritized_memory import Memory as PrioritizedMemory
import os

if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    NUM_ACTIONS = 2
    DISCOUNT = 0.99
    STATE_SHAPE = (4,)

    #specify function approximators
    qnets = networks.SimpleDueling((10,10), STATE_SHAPE, NUM_ACTIONS)

    policy = networks.SimpleDense((10,10))
    policy.add(tf.keras.layers.Dense(NUM_ACTIONS))
    policy.add(tf.keras.layers.Softmax())

    #1
    algorithm = MarkovDuelingDiscreteSAC(policy, qnets, STATE_SHAPE, NUM_ACTIONS, 0.99)

    #2
    agent = PolicyAgent(algorithm.policy)

    #3
    state_manager = MarkovStateManager(agent, env)

    #4
    actor = BasicActor(state_manager)

    #5
    memory = PrioritizedMemory(int(1e6))

    #6
    learner = PrioritizedLearner(algorithm, 1000, 64, log_every=5).train_undistributed(actor, memory)




