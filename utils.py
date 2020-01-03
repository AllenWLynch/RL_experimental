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

class EnvWrapper():

    def __init__(self, env, reward_scale = 1.0, state_lag = 1):
        self.reward_scale = reward_scale
        self.env = env
        self.state_shape = np.array(env.reset()).shape
        self.state_lag = state_lag
        self.reset()

    def prep_state(self, new_state):
        self.state_buffer.insert(0, new_state)
        self.state_buffer.pop()
        return tf.concat(self.state_buffer, axis = -1)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        return (self.prep_state(state), reward * self.reward_scale, done, info)

    def reset(self):
        self.state_buffer = [np.zeros(self.state_shape) for i in range(self.state_lag)]
        return self.prep_state(self.env.reset())
    
    def render(self):
        self.env.render()

    def get_state_shape(self):
        return tf.concat(self.state_buffer, axis = -1).get_shape()

class DiscreteEpisodicRL():

    def __init__(self, algo, replay_buffer, logdir):
        self.algo = algo
        self.replay = replay_buffer
        self.train_steps = 0
        self.episodes = 0
        self.logger = tf.summary.create_file_writer(logdir)

    def train_episode(self, env, batch_size = 64, t_max = np.inf, render = False):

            state = env.reset()
            done = False
            t = 0
            rewards = []
            while not done and t < t_max:
                
                #sample step
                action = self.algo.get_action(state)
                
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

    def train(self, env, epochs, episodes_per_epoch, batch_size = 64, t_max = np.inf, render = False):

        for epoch in range(1, epochs + 1):

            print('Epoch {}'.format(str(epoch)))
            for episode in range(1, episodes_per_epoch + 1):

                self.train_episode(env, batch_size, t_max, render=render)
                print('\rEpisode: {}'.format(str(episode)), end = '')
                
            print('Epoch {} complete, evaluating results.'.format(str(epoch)))