import gym

#print(gym.envs.registry.all())

env = gym.make('Acrobot-v1')

print(env.action_space)

#print(env.action_space.high, env.action_space.low)

print(env.observation_space)