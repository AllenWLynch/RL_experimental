
import tensorflow as tf
import numpy as np
import gym
import networks
from SAC_algos import DuelingDiscreteSAC
from utils import Replay, DiscreteEpisodicRL, EnvWrapper, concat_time_series, one_hot_action_in_state


env = gym.make('CartPole-v1')

NUM_ACTIONS = 2
DISCOUNT = 0.99

def gen_state(shapes):
    (state_shape, action_shape) = shapes
    return np.zeros(state_shape), np.zeros(action_shape)

def concat_states_actions(x):
    states, actions = list(zip(*x))
    return np.concatenate([*states, actions[-1]], axis = -1)

env_wrapper = EnvWrapper(env, gen_state, reward_scale = 0.01, state_lag = 10, 
    state_processing_function = one_hot_action_in_state(NUM_ACTIONS), output_processing_fn = concat_states_actions)

state_shape = env_wrapper.get_state_shape()

q_net = networks.SimpleDueling((50,50),state_shape,NUM_ACTIONS)

policy_net = networks.SimpleDense((50,50))
policy_net.add(tf.keras.layers.Dense(NUM_ACTIONS))
policy_net.add(tf.keras.layers.Softmax())

#algo = DiscreteSACwithValueNet(q_net, policy_net, v_net, tf.keras.optimizers.Adam, 3e-4, 0.05, DISCOUNT, state_shape, soft_update_beta=0.995)

algo = DuelingDiscreteSAC(
                policy_net,
                q_net,
                state_shape, 
                NUM_ACTIONS)

trainer = DiscreteEpisodicRL(algo, Replay(1e6), './logs')

trainer.train(env_wrapper, 1, 1000, render = True, initial_random_steps=0)