import numpy as np
from collections import namedtuple
import random
import tensorflow as tf
import threading
import datetime
import os

class RL_Algo():

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def update_step(self, *args):
        raise NotImplementedError()

def unzip_batch_samples(batch):
    return map(lambda x : np.array(x), list(zip(*batch)))

def calculate_priority(td_errors, nu):
    # p = η maxi δi + (1 − η)δ-
    assert(len(td_errors.shape) == 2), 'td_errors must be of shape (m, n), where m = batch_size, n = sequence length'
    return nu * np.max(td_errors, axis = -1, keepdims= True) + (1 - nu) * np.mean(td_errors, axis = -1, keepdims= True)

class NaiveReplay():

    def __init__(self, size):
        self.size = size
        self.memory = []

    def add(self, sample):
        if len(self.memory) > self.size:
            self.memory.pop()
        self.memory.insert(0, sample)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), np.random.rand(batch_size, 1), 1.0

    def num_samples(self):
        return len(self.memory)

    def update(self, *args):
        pass

#remake this
def get_n_step_Q_targets(reward_sequence, lookahead_steps, done, discount, last_V_estimate):

    (m, N) = reward_sequence.shape

    assert(done.shape == last_V_estimate.shape), 'Done and last_V_estimate must have same dimensions: (m, 1)'
    assert(done[0] == m)
    assert(done[1] == 1)
    
    last_V_estimate *= (1 - done)

    reward_sequence = np.expand_dims(np.concatenate([reward_sequence, last_V_estimate], axis = -1), 1)

    upper_tri = np.tri(N + 1).T

    R = reward_sequence * upper_tri

    k = np.arange(N + 1)[:, np.newaxis]

    K = k.T - k

    discount_matrix = discount ** K

    bootstrapped_rewards = R * discount_matrix #element-wise

    reward_sums = np.sum(bootstrapped_rewards, axis = -1, keepdims = True)

    return np.squeeze(reward_sums)[:,:-1]


InteractionSequence = namedtuple('InteractionSequence', ('states','actions','rewards','is_state','done'))
MarkovTransition = namedtuple('MarkovTransition', ('state','action','reward','next_state', 'done'))

class Episode():

    def __init__(self):
        self.returns = 0
        self.steps = 0
        self.terminal = 0

    def track(self, reward, done):
        self.returns += reward
        self.steps += 1
        if done:
            self.terminal = True

    def result(self):
        return (self.returns, self.returns/self.steps), ('Total returns', 'Returns per step')


class MarkovStateManager():
    
    def __init__(self, agent, simulator, render = True,
        max_episode_len = np.inf, preprocessing_function = lambda x, a : x, reward_fn = lambda x : x):

        self.env = simulator
        self.reward_fn = reward_fn
        self.phi = preprocessing_function
        self.agent = agent
        self.render = render

        self.max_t = max_episode_len
        self.t = 0

        self.state_shape = self.infer_state_shape(self.phi(self.env.reset(), None))
        #blank_state_shape = self.infer_state_shape(self.state_generator())
    
        #assert(example_state_shape == blank_state_shape), 'States from preprocessing funtion and blank state generator must have the same shape: {} vs. {}'.format(str(example_state_shape), str(blank_state_generator))
        self.reset()
        
    def get_state_shape(self):
        return self.state_shape

    def reset(self):

        self.t = 0
        #self.interaction_history = [Interaction(self.state_generator(), 0, 0, False) for i in range(self.seq_len)]
        self.state = self.phi(self.env.reset(), None)
        self.episode = Episode()
        #self.save_memory(self.state, 0, 0)

    def infer_state_shape(self, state):
        #print(state)
        if type(state) == tuple:
            #print('a')
            return tuple([self.infer_state_shape(x) for x in state])
        elif type(state) == np.ndarray:
            #print('b')
            return state.shape
        elif type(state) == list:
            #print('c')
            return (len(state), self.infer_state_shape(state[0]))
        else:
            raise TypeError('State must contain only tuples, lists, or numpy arrays')

    def run(self):

        self.reset()

        while True:

            action = self.agent.get_action(self.state)

            self.t += 1

            next_state, reward, done, _ = self.env.step(action)
            next_state = self.phi(next_state, action)
            reward = self.reward_fn(reward)

            if self.render:
                self.env.render()

            self.episode.track(reward, done)

            yield MarkovTransition(self.state, [action], [reward], next_state, [done]), self.episode.result() if self.episode.terminal else False

            self.state = next_state

            if done or self.t >= self.max_t:
                self.reset()

class SequentialStateManager(MarkovStateManager):

    def __init__(self, agent, simulator, blank_state_generator, render = True, seq_len = 40, overlap = 0,
                max_episode_len = np.inf, preprocessing_function = lambda x, a : x, reward_fn = lambda x : x):

        self.env = simulator
        self.reward_fn = reward_fn
        self.phi = preprocessing_function
        self.state_generator = blank_state_generator
        self.seq_len = seq_len
        self.overlap = overlap
        self.agent = agent
        self.render = render

        self.max_t = max_episode_len
        self.t = 0

        example_state_shape = self.infer_state_shape(self.phi(self.env.reset(), None))
        blank_state_shape = self.infer_state_shape(self.state_generator())
        
        assert(example_state_shape == blank_state_shape), 'States from preprocessing funtion and blank state generator must have the same shape: {} vs. {}'.format(str(example_state_shape), str(blank_state_generator))
        self.reset()
        
    def push_interaction(self, state, action, reward):
        for newval, history in zip((state, action, reward, True), (self.state_history, self.action_history, self.reward_history, self.is_state)):
            history.insert(0, newval)
            del history[-1]

    def get_state_shape(self):
        return self.infer_state_shape(self.get_policy_history())

    def get_policy_history(self):
        return np.array(self.state_history)

    def reset(self):

        self.t = 0

        self.state_history = [self.state_generator() for i in range(self.seq_len)]
        self.action_history = [0 for i in range(self.seq_len)]
        self.reward_history = [0 for i in range(self.seq_len)]
        self.is_state = [False for i in range(self.seq_len)]
        
        self.state = self.phi(self.env.reset(), None)

    def run(self):

        while True:

            action = self.agent.get_action(np.array(self.state_history).astype('float32'), np.array(self.is_state).astype('bool'))
            
            self.t += 1

            next_state, reward, done, _ = self.env.step(action)
            next_state = self.phi(next_state, action)
            reward = self.reward_fn(reward)
            self.push_interaction(self.state, action, reward)

            self.state = next_state

            if self.render:
                self.env.render()

            if done or self.t % (self.seq_len - self.overlap) == 0 or self.t >= self.max_t:
                print(self.t)
                yield InteractionSequence(
                        np.array(self.state_history).astype('float32'), 
                        np.array(self.action_history).astype('int32'), 
                        np.array(self.reward_history).astype('float32'),
                        np.array(self.is_state).astype('bool'),
                        np.array([done]).astype('bool'),
                        )
                if done or self.t > self.max_t:
                    self.reset()

class PrioritizedLearner():

    def __init__(self, rl_algo, steps_per_epoch, batch_size, priority_nu = 1.0, checkpoint_every = 10, log_every = 50, logdir = 'logs'):

        self.algo = rl_algo
        self.train_steps = 0
        self.checkpoint_every = checkpoint_every
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.nu = priority_nu
        self.logger = tf.summary.create_file_writer(
            os.path.join(
                logdir,
                "fit/",
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            ))
        self.log_every = log_every
        self.episodes = 0

    def update_step(self, prioritized_replay_sample):

        self.train_steps += 1
        if self.train_steps % self.steps_per_epoch == 0:
            epoch_num = self.train_steps // self.steps_per_epoch
            print('\nEpoch {}'.format(epoch_num))
            if epoch_num % self.checkpoint_every == 0:
                print('Saving Checkpoint!')

        print('\rTrain step: {}'.format(str(self.train_steps)), end = '')
        
        (batch, idx, importance_weight) = prioritized_replay_sample

        importance_weight = np.array(importance_weight).reshape((-1,1)).astype('float32')

        td_errors, vals, metric_names = self.algo.update_step(*unzip_batch_samples(batch), weights = importance_weight)

        td_errors = calculate_priority(td_errors, self.nu)

        if self.train_steps % self.log_every == 0:
            with self.logger.as_default():
                for (val, metric) in zip(vals, metric_names):
                    tf.summary.scalar(metric, val, step = self.train_steps)

        return idx, td_errors.reshape(-1)

    def log_episode(self, vals, metric_names):
        with self.logger.as_default():
            for (val, metric) in zip(vals, metric_names):
                tf.summary.scalar(metric, val, step = self.episodes)
        self.episodes += 1

    def train_undistributed(self, actor, memory, env_steps = 1):
        actor_iter = iter(actor.run())

        while True:
            for i in range(env_steps):
                transition, episode = next(actor_iter)
                memory.add(transition)
                if not (episode is False):
                    self.log_episode(*episode)
            if memory.num_samples() > self.batch_size:
                for idx, error in zip(*self.update_step(memory.sample(self.batch_size))):
                    memory.update(idx, error)

    
