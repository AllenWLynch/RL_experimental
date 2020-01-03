import tensorflow as tf
import numpy as np
from utils import RL_Algo

class DiscreteSAC(RL_Algo):

    def __init__(self, 
                networks, 
                optims,  
                entrop_weight,
                discount,
                state_shape,
                soft_update_beta = 0.99):
        # init parameters
        assert(len(networks)), 'Networks parameter must contain three networks: (policy, q_fn1, q_fn2, value_fn)'
        (self.policy, self.q1, self.q2, self.value_net) = networks

        input_shape = (None, *state_shape)
        self.policy.build(input_shape)
        self.q1.build(input_shape)
        self.q2.build(input_shape)
        self.value_net.build(input_shape)

        self.value_target = tf.keras.models.clone_model(self.value_net)
        self.value_target.build(input_shape)
        self.value_target.set_weights(self.value_net.get_weights())

        (self.policy_opt, self.q1_opt, self.q2_opt, self.v_opt) = optims
        self.entrop_weight = entrop_weight
        self.discount = discount
        #self.num_actions = num_actions
        self.soft_update_beta = soft_update_beta
        #self.replay = Replay(replay_size)

    def q_loss(self, q_val, action, target):
        #MSE, average across batch dimension
        td_error = tf.gather(q_val, action, axis = -1, batch_dims = 1) - target
        #print('TD error shape: ', td_error.get_shape())
        return tf.reduce_mean(tf.square(td_error))

    def value_loss(self, state_val, target):
        #MSE of td error, averaged across batch_dimension
        return tf.reduce_mean(tf.square(state_val - target))

    def expectation_over_actions(self, action_distribution, q_val):
        expectation = tf.reduce_sum(tf.multiply(action_distribution, q_val - self.entrop_weight * tf.math.log(action_distribution)),axis = -1,keepdims = True)
        return expectation

    def update_step(self, s, a , r, s_next, done):

        with tf.GradientTape(persistent = True) as tape:

            q_target = r + self.discount * (1. - tf.dtypes.cast(done, 'float32')) * self.value_target(s_next)

            q1_vals, q2_vals = self.q1(s), self.q2(s)

            compare_qs = tf.stack([q1_vals, q2_vals])

            min_q = tf.reduce_min(compare_qs, axis = 0)

            action_distribution = self.policy(s)

            v_target = self.expectation_over_actions(action_distribution, min_q)

            #use targets to compute losses
            q1_loss = self.q_loss(q1_vals, a, q_target)
            q2_loss = self.q_loss(q2_vals, a, q_target)

            v_loss = self.value_loss(self.value_net(s), v_target)

            #want to minimize the negative of the maximization objective
            policy_loss = -1 * tf.reduce_mean(self.expectation_over_actions(action_distribution, q1_vals))

        q1_grads, q2_grads = tape.gradient(q1_loss, self.q1.trainable_weights), tape.gradient(q2_loss, self.q2.trainable_weights)
        v_grads = tape.gradient(v_loss, self.value_net.trainable_weights)
        policy_grads = tape.gradient(policy_loss, self.policy.trainable_weights)

        del tape

        self.q1_opt.apply_gradients(zip(q1_grads, self.q1.trainable_weights))
        self.q2_opt.apply_gradients(zip(q2_grads, self.q2.trainable_weights))
        self.policy_opt.apply_gradients(zip(policy_grads, self.policy.trainable_weights))
        self.v_opt.apply_gradients(zip(v_grads, self.value_net.trainable_weights))

        self.value_target.set_weights([
            self.soft_update_beta * targ_weight + (1.0 - self.soft_update_beta) * update_weights
            for targ_weight, update_weights in zip(self.value_target.get_weights(), self.value_net.get_weights())
        ])

        return (q1_loss, q2_loss, v_loss, policy_loss), ('Q1 loss', 'Q2 loss', 'Value loss','Policy loss')

    def get_action(self, state):
        distribution = (self.policy( np.array([state]) )[0]).numpy()
        return np.random.choice(len(distribution), p = distribution)
        

        



















        