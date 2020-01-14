from collections import namedtuple
import numpy as np
import random
import tensorflow as tf

Transition = namedtuple('Transition', ('state','action','reward','next_state', 'done'))

class Replay():

    def __init__(self, size):
        self.size = size
        self.memory = []

    def add(self, *args):
        if len(self.memory) > self.size:
            self.memory.pop()
        self.memory.insert(0, Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def num_samples(self):
        return len(self.memory)

class RL_Algo():

    def __init__(self):
        raise NotImplementedError()

    def update_step(self):
        raise NotImplementedError()

    def get_action_distribution(self):
        raise NotImplementedError()


def unmodified_state_fn(**kwargs):
    return kwargs['state']

def concat_time_series(state):
    return tf.concat(state, axis = -1)

class one_hot_action_in_state():

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def __call__(self, **kwargs):
        state = kwargs['state']
        action = np.zeros(self.num_actions)
        if 'action' in kwargs:
            action[kwargs['action']] = 1
        return (state, action)

class EnvWrapper():

    def __init__(self, env, state_generator_fn,reward_scale = 1.0, state_lag = 1,
                state_processing_function = unmodified_state_fn, output_processing_fn = lambda x : x):
        self.reward_scale = reward_scale
        self.env = env
        self.processing_fn = state_processing_function
        self.state_generator = state_generator_fn

        start_state = self.processing_fn(state = env.reset())
        self.state_shape = self.infer_state_shape(start_state)

        self.state_lag = state_lag
        self.output_fn = output_processing_fn
        self.reset()
        self.action_space = env.action_space

    def push_state(self, new_state):
        self.state_buffer.insert(0, new_state)
        self.state_buffer.pop()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = self.processing_fn(state = state, action = action)
        self.push_state(state)
        return (self.output_fn(self.state_buffer), reward * self.reward_scale, done, info)

    def reset(self):
        start_state = self.processing_fn(state = self.env.reset())
        self.state_buffer = [self.state_generator(self.state_shape) for i in range(self.state_lag)]
        self.push_state(start_state)
        return self.output_fn(self.state_buffer)

    def render(self):
        self.env.render()

    def get_state_shape(self):
        return self.infer_state_shape(self.output_fn(self.state_buffer))

    #can be composed of tuples or ndarrays
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
        
class DiscreteEpisodicRL():

    def __init__(self, algo, replay_buffer, logdir):
        self.algo = algo
        self.replay = replay_buffer
        self.train_steps = 0
        self.episodes = 0
        self.logger = tf.summary.create_file_writer(logdir)

    def train_episode(self, env, random_threshold, batch_size = 64, t_max = np.inf, render = False):

            state = env.reset()      

            done = False
            t = 0
            rewards = []

            while not done and t < t_max:
                
                #sample step
                if self.train_steps > random_threshold:
                    action = self.algo.get_action(state)
                else:
                    action = env.action_space.sample()
                
                state_next, reward, done, _ = env.step(action)

                if render:
                    env.render()

                self.replay.add(state, [action], [reward], state_next, [done])

                state = state_next

                rewards.append(reward)

                #update step
                if self.replay.num_samples() > batch_size:
                
                    memory_sample = self.replay.sample(batch_size)

                    s, a, r, s_next, dones = [tf.stack(sample) for sample in zip(*memory_sample)]

                    vals, metric_names = self.algo.update_step(s, a, r, s_next, dones)

                    with self.logger.as_default():
                        for (val, metric) in zip(vals, metric_names):
                            tf.summary.scalar(metric, val, step = self.train_steps)

                t += 1
                self.train_steps += 1

            self.episodes += 1

            with self.logger.as_default():
                tf.summary.scalar('Returns', sum(rewards), step = self.episodes)

    def train(self, env, epochs, episodes_per_epoch, batch_size = 64, t_max = np.inf, render = False, initial_random_steps = 1e4):

        for epoch in range(1, epochs + 1):

            print('Epoch {}'.format(str(epoch)))
            for episode in range(1, episodes_per_epoch + 1):

                self.train_episode(env, initial_random_steps, batch_size, t_max, render=render)
                print('\rEpisode: {}'.format(str(episode)), end = '')
                
            print('Epoch {} complete, evaluating results.'.format(str(epoch)))