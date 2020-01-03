
import tensorflow as tf
import numpy as np
import gym
import networks
from SAC_algos import DiscreteSAC
from utils import Replay, DiscreteEpisodicRL, EnvWrapper


env = gym.make('Acrobot-v1')

NUM_ACTIONS = 3
DISCOUNT = 0.99

env_wrapper = EnvWrapper(env, 0.1, 4)

q1 = networks.SimpleDense((50,50))
q1.add(tf.keras.layers.Dense(NUM_ACTIONS, activation = 'linear'))

q2 = networks.SimpleDense((50,50))
q2.add(tf.keras.layers.Dense(NUM_ACTIONS, activation = 'linear'))

v = networks.SimpleDense((50,50))
v.add(tf.keras.layers.Dense(1, activation = 'linear'))

policy_net = networks.SimpleDense((50,50))
policy_net.add(tf.keras.layers.Dense(NUM_ACTIONS))
policy_net.add(tf.keras.layers.Softmax())

state_shape = env_wrapper.get_state_shape()

print('State shape:', state_shape)

algo = DiscreteSAC((policy_net, q1, q2, v), 
                [tf.keras.optimizers.Adam(3e-4) for i in range(4)],  
                1.0,
                DISCOUNT,
                state_shape,
                )

trainer = DiscreteEpisodicRL(algo, Replay(1e6), './logs')

trainer.train(env_wrapper, 1, 1000, render = True)

