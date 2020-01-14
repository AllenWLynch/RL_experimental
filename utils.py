import numpy as np
from collections import namedtuple
import random
import tensorflow as tf

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
        self.num_actions = 2

    def get_action(self, state, state_mask = None):
        return int(np.random.choice(np.arange(self.num_actions)))

class PolicyAgent(Agent):

    def __init__(self, policy, noise = 0):
        self.policy = policy
        self.noise = noise
        #self.num_actions = num_actions

    def get_action(self, state, state_mask = None):
        distribution = (self.policy(np.expand_dims(state, 0))[0]).numpy()
        if self.noise > 0:
            raise NotImplementedError()

        return np.random.choice(len(distribution), p = distribution)

class NaiveReplay():

    def __init__(self, size):
        self.size = size
        self.memory = []

    def add(self, *args):
        if len(self.memory) > self.size:
            self.memory.pop()
        self.memory.insert(0, MarkovTransition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def num_samples(self):
        return len(self.memory)

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

'''r = np.array([[1,2,3,4],[2,4,6,8]])
V = np.array([[5],[10]])
done = np.array([[True],[False]])
discount = 0.5
print(get_Q_targets(r, done, discount, V))'''

def unzip_batch_samples(batch):
    return map(lambda x : np.array(x).astype('float32'), list(zip(*batch)))

def calculate_n_step_priority(td_errors, nu):
    # p = η maxi δi + (1 − η)δ-
    assert(len(td_errors.shape) == 2), 'td_errors must be of shape (m, n), where m = batch_size, n = sequence length'
    return nu * np.max(td_errors, axis = -1, keepdims= True) + (1 - nu) * np.mean(td_errors, axis = -1, keepdims= True)

InteractionSequence = namedtuple('InteractionSequence', ('states','actions','rewards','is_state','done'))

MarkovTransition = namedtuple('MarkovTransition', ('state','action','reward','next_state', 'done'))

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

            #self.save_memory(self.state, action, reward, done)
            if self.render:
                self.env.render()

            yield MarkovTransition(self.state, [action], [reward], next_state, [done])

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
                        np.array(self.action_history).astype('float32'), 
                        np.array(self.reward_history).astype('float32'),
                        np.array(self.is_state).astype('bool'),
                        np.array([done]).astype('bool'),
                        )
                if done or self.t > self.max_t:
                    self.reset()

class LocalSynchronousActor():

    def __init__(self, state_manager, memory):
        self.state_manager = state_manager
        self.memory = memory

    def run(self):
        for transition in self.state_manager.run():
            self.memory.add(transition)
            yield

class PrioritizedLocalSynchronousLearner():

    def __init__(self, rl_algo, memory, steps_per_epoch, epochs = 100, checkpoint_every = 10, logdir = './logs'):

        self.algo = rl_algo
        self.memory = memory
        self.logger = tf.summary.create_file_writer(logdir)
        self.train_steps = 0
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

    def train_step(self, batch_size):

        if self.memory.num_samples() > batch_size:
                
            memory_sample = self.memory.sample(batch_size)

            #s, a, r, s_next, dones = [tf.stack(sample) for sample in zip(*memory_sample)]

            td_errors, vals, metric_names = self.algo.update_step(memory_sample)

            #update memory
            #

            with self.logger.as_default():
                for (val, metric) in zip(vals, metric_names):
                    tf.summary.scalar(metric, val, step = self.train_steps)

        self.train_steps += 1


    


class DiscreteEpisodicRL():

    def __init__(self, algo, replay_buffer, logdir):
        self.algo = algo
        self.replay = replay_buffer
        
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











