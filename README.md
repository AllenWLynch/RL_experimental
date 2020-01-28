
# Soft Actor-Critic

## Description
Soft actor-critic is a SOTA reinforcement learning algorithm that modifies the classic reinforcement learning objective -expected return maximization- with an added entropy term that encourages the policy to maintain a hyperparameterized degree of randomness. This objective balances exploration and exploitation depending on what has been learned about a given state, and provides smoother transitions between action determinations while learning compared to epsilon-greedy policies, which may jarringly switch the action taken in a given state when the maximum Q-value changes. Besides the entropy modification, SAC includes other recent advancements in DQN frameworks to improve training stability:

* Jointly learns policy and state-action values for faster convergeance on solution
* Learns off policy for maximum sampling efficiency
* Utilizes clipped double-Q, similar to TD3
* Soft updates of target networks, rather than scheduled updates

One update step of SAC, shown below, depends on a stored (s, a, r, s') transition to train two Q-networks and a policy network. Entropy in the policy network is maintained with a temperature parameter alpha, which is also learnable. The two Q-target networks are updated by Polyak averaging at every step with the true Q-networks. Gradients are shown with dashes.

<img src='readme_materials/sac_diagram.png' height="512">
<br>Figure 1. SAC algorithm

My implementation of SAC is modified to work with discrete action spaces and Dueling Q-networks. The authors of the SAC paper state that the use of a dedicated target value network improves the stability of training, but I hypothesize this value network can be eliminated with the use of Dueling Q-networks, also implemented in this repo, which include a value stream as part of their formulation. 

I plan to combine the SAC framework with distributed prioritized replay (Fig. 3), recently shown to vastly improve training speed of DeepRL agents, and recent advancements in sequence-processing neural networks to create a SOTA reinforcement learing algorithm for discrete action spaces. Namely, I plan to implement a new style of Q and policy networks that moves away from the stacked-frames convolutional and recurrent mechanisms used in the past to plan from a series of past states. Instead, I will use the self-attention mechanism found in transformer models, which are faster and have longer memory than recurrent mechansisms. Additionally, I plan to maintain the volume-dimensionality of spatially-distributed image data in my Transformer model, rather than embedding the image state into a 1D vector. Maintaining spatial structure may aid in deriving value from memories of past states (Fig. 2). I believe the combination of SAC, distributed prioritized replay, state-transformer networks, and dueling Q-networks will produce a new SOTA benchmark.

<img src="readme_materials/state_transformer.png">
<br>Figure 2. State transformer model.<br><br>

<img src="readme_materials/actor-learner.png">
<br>Figure 3. Distributed prioritized experience replay with Actor-Learner framework. The Learner module may be duplicated many times to increase rate of experience.

## Progress

I have implemented and trained SAC agents in sequential and markovian environments with prioritized replay. Next, I must finish setting up the distributed Actor-Memory-Learner architecture by writing a threadsafe memory process. Following this, I will compare training of LSTM and Transformer models in Atari Arcade environments.

Currently, I have results showing the improvement on total returns per episode in the Cartpole-v1 environment due to the introduction of prioritized replay. Both runs were trained using the SAC algorithm with the same hyperparameters.

<br>
<img src="readme_materials/prioritized_replay_effects.png" alt="Prioritized vs Niave Replay" height="370">
<br>Figure 4. Total returns per episode with and without prioritized experience replay.


## Framework

The code below shows the setup necessary to run the tests in Figure 4. My framework splits the task of specifying and training an RL agent into discrete systems. 
1.   The "Algorithm" holds the function approximator or tabular datastructure used to learn how to maximize rewards in the environment. This object is passed transition samples from which it updates its values.
2.   The "Agent" interacts with the environment. It may choose actions based on policy or value estimates made by the algorithm, follow an epsilon greedy policy, act randomly, etc.
3.   The "State Manager" wraps an environment and agent and manages reward scaling, state pre-processing, or past-state tracking for sequential control. 
4.   The "Actor" is a higher level term that packages a State Manager into a bundle that may be distributed. The State Manager calls the agent for actions.
5.   The "Memory" contains transition states for off-policy algorithms like SAC to sample. Actors contribute to new memories.
6.   Finally, the "Learner" specifies a training routine for the Algorithm, and coordinates the three discrete components (Learner, Actor, Memory) to execute the training.

```python
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
```
## References

1. <a href="https://openreview.net/pdf?id=H1Dy---0Z">DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY</a>
2. <a href="https://openreview.net/forum?id=r1lyTjAqYX">Recurrent Experience Replay in Distributed Reinforcement Learning</a>
3. <a href="https://arxiv.org/abs/1801.01290">Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor</a>
4. <a href="https://arxiv.org/pdf/1910.07207.pdf">SOFT ACTOR-CRITIC FOR DISCRETE ACTION SETTINGS</a>