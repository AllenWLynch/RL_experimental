import gym
from utils import MarkovStateManager, RandomAgent, SequentialStateManager, PrioritizedLearner, PrioritizedPolicyAgent
import numpy as np
from SAC_algos import MarkovDuelingDiscreteSAC
from prioritized_memory import Memory
import networks
import tensorflow as tf
from parallel_utils import ParallelActor
import threading
import multiprocessing
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def add_to_memory_fn(batch, td_errors):
    print('Batch added to memory!')

def get_models(state_shape, num_actions):

    q_net = networks.SimpleDueling((50,50), state_shape, num_actions)

    policy_net = networks.SimpleDense((50,50))
    policy_net.add(tf.keras.layers.Dense(num_actions))
    policy_net.add(tf.keras.layers.Softmax())

    return policy_net, q_net


def run_remote_actor(env, memory_fn, *model_args, **manager_kwargs):
    
    state_shape, num_actions, discount = model_args

    input_shape = (None, *state_shape)

    with tf.device("/device:CPU:0"):
        
        pol, q = get_models(state_shape, num_actions)

        q.build(input_shape)
        q2 = tf.keras.models.clone_model(q)
        q2.build(input_shape)
        pol.build(input_shape)

        agent = PrioritizedPolicyAgent(pol, q, q2, discount)

        manager = MarkovStateManager(agent, env, **manager_kwargs)

        actor = ParallelActor(manager, memory_fn)

        actor.start()


if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    NUM_ACTIONS = 2
    DISCOUNT = 0.99
    STATE_SHAPE = (4,)

    algorithm = MarkovDuelingDiscreteSAC(*get_models(STATE_SHAPE, NUM_ACTIONS), STATE_SHAPE, NUM_ACTIONS, DISCOUNT)

    p = multiprocessing.Process(target = run_remote_actor, 
        args = (env, add_to_memory_fn, STATE_SHAPE, NUM_ACTIONS, DISCOUNT), #policy_net, algorithm.q1, algorithm.q1_targ, DISCOUNT),
        kwargs= {'render' : True})
    p.daemon= True
    p.start()

    while True:
        print('Main loop')
        time.sleep(1)
