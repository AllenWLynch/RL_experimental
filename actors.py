import threading
import queue
import numpy as np
import tensorflow as tf

#this parallel actor will sit on one process/CPU core, but will thread multiple agents, is assisted in exploration and diversification with noisy policy
class PriorityActor():

    def __init__(self, state_manager, memory_addition_function, buffer_size = 64):
        
        self.manager = state_manager
        self.buffer_size = 64
        self.buffer = queue.Queue()
        self.memory_addition_function = memory_addition_function

    def process_batch(self):
        if not self.buffer.empty():
            next_batch = self.buffer.get()
            td_errors = self.manager.agent.estimate_batch_priorities(next_batch)
            self.memory_addition_function(next_batch, td_errors)

    def start(self):

        iterator = iter(self.manager.run())
        while True:
            batch = [
                next(iterator) for j in range(self.buffer_size)
            ]
            self.buffer.put(batch)
            if t1:
                t1.join()
            t1 = threading.Thread(target = self.process_batch)
            t1.start()

class BasicActor():

    def __init__(self, state_manager):
        self._wrapped_manager = state_manager

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_manager, attr)

#__AGENTS___

class Agent():

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def get_action(self, state):
        raise NotImplementedError()

class SequentialAgent(Agent):

    def get_action(self, state, is_state):
        raise NotImplementedError()

class RandomAgent(Agent):

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def get_action(self, state, state_mask = None):
        return int(np.random.choice(np.arange(self.num_actions)))

class PolicyAgent(Agent):

    def __init__(self, policy, noise = 0):
        #assert(False), 'Dont use this one use the prioritized guy'
        self.policy = policy
        self.noise = noise
        #self.num_actions = num_actions

    def get_action(self, state, state_mask = None):
        policy_output = self.policy(np.expand_dims(state, 0))[0]
        distribution = (policy_output).numpy()
        if self.noise > 0:
            raise NotImplementedError()

        return np.random.choice(len(distribution), p = distribution)

class PrioritizedPolicyAgent(PolicyAgent):

    def __init__(self, policy, q_net, q_targ, discount):
        self.policy = policy
        self.q_net = q_net
        self.q_targ = q_targ
        self.discount = discount
        self.noise = 0

    def estimate_batch_priorities(self, batch):
        state, action, rewards, next_states, dones = unzip_batch_samples(batch)

        q_vals = tf.gather(self.q_net(state), action, axis = -1, batch_dims = 1)

        next_qs = self.q_targ(next_states)
        max_next_action = tf.math.argmax(next_qs, axis = -1)
        
        max_next_action = tf.reshape(max_next_action, (-1,1))

        v_estimates = tf.gather(next_qs, max_next_action, axis = -1, batch_dims = 1)

        q_target = rewards + self.discount * (1. - tf.dtypes.cast(dones, 'float32')) * v_estimates

        return tf.math.abs(q_vals - q_target).numpy().reshape(-1) 





