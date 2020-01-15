import tensorflow as tf
import numpy as np
from utils import RL_Algo

class MarkovDuelingDiscreteSAC(RL_Algo):

    def __init__(self, 
                policy_architecture,
                q_architecture, 
                state_shape,
                num_actions,
                optimizer = tf.keras.optimizers.Adam,
                learning_rate = 0.0003,
                soft_update_beta = 0.995,
                discount = 0.99,
                initial_entropy = 1.0):
        # init parameters
        input_shape = (None, *state_shape)
        
        self.policy = policy_architecture
        self.policy.build(input_shape)
        self.policy_opt = optimizer(learning_rate)

        self.q1, self.q2, self.q1_targ, self.q2_targ = [tf.keras.models.clone_model(q_architecture) for i in range(4)]

        #bulids the four q-nets
        self.q1.build(input_shape)
        self.q1_targ.build(input_shape)
        self.q1_targ.set_weights(self.q1.get_weights())
        self.q1_opt = optimizer(learning_rate)

        self.q2.build(input_shape)
        self.q2_targ.build(input_shape)
        self.q2_targ.set_weights(self.q2.get_weights())
        self.q2_opt = optimizer(learning_rate)

        self.entropy_target = -1 * num_actions

        self.discount = discount
        self.soft_update_beta = soft_update_beta
        self.entropy_weight = tf.Variable(initial_entropy, trainable = True)
        self.entropy_opt = optimizer(learning_rate)

    def update_targ_network(self, targ_network, new_weights):
        targ_network.set_weights([
            self.soft_update_beta * targ_weight + (1.0 - self.soft_update_beta) * update_weights
            for targ_weight, update_weights in zip(targ_network.get_weights(), new_weights)
        ])

    def q_loss(self, q_val, action, target):
        #MSE, average across batch dimension
        td_error = tf.gather(q_val, action, axis = -1, batch_dims = 1) - target
        #print('TD error shape: ', td_error.get_shape())
        return tf.square(td_error), tf.math.abs(td_error)

    def expectation_over_actions(self, action_distribution, q_val):
        expectation = tf.reduce_sum(tf.multiply(action_distribution, q_val - self.entropy_weight * tf.math.log(action_distribution)),axis = -1,keepdims = True)
        return expectation

    def update_step(self, s, a , r, s_next, done, weights = 1.0):

        q1_targs, q2_targs = self.q1_targ(s_next), self.q2_targ(s_next)

        min_q = tf.reduce_min(tf.stack([q1_targs, q2_targs]), axis = 0)

        next_state_distribution = self.policy(s_next)

        v_estimate = self.expectation_over_actions(next_state_distribution, min_q)

        q_target = r + self.discount * (1. - tf.dtypes.cast(done, 'float32')) * v_estimate

        weights = tf.Variable(weights, trainable = False)

        with tf.GradientTape(persistent = True) as tape:

            #tape.watch(weights)

            action_distribution = self.policy(s)

            with tf.GradientTape(watch_accessed_variables=False) as entropy_tape:
                entropy_tape.watch(self.entropy_weight)
                entropy_loss = tf.reduce_sum(
                    tf.multiply(action_distribution, -self.entropy_weight * (tf.math.log(action_distribution) + self.entropy_weight)), 
                    -1, keepdims = True)
                mean_entropy_loss = tf.reduce_mean(weights * entropy_loss)

            q1_vals, q2_vals = self.q1(s), self.q2(s)
            #use targets to compute losses
            q1_loss, td_error1 = self.q_loss(q1_vals, a, q_target)
            q2_loss, td_error2 = self.q_loss(q2_vals, a, q_target)
            #want to minimize the negative of the maximization objective
            policy_loss = -1 * tf.reduce_mean(weights * self.expectation_over_actions(action_distribution, q1_vals))

            q1_loss = tf.reduce_mean(weights * q1_loss)
            q2_loss = tf.reduce_mean(weights * q2_loss)

        q1_grads = tape.gradient(q1_loss, self.q1.trainable_weights)
        q2_grads = tape.gradient(q2_loss, self.q2.trainable_weights)
        policy_grads = tape.gradient(policy_loss, self.policy.trainable_weights)
        entropy_grad = tape.gradient(mean_entropy_loss, self.entropy_weight)

        del tape

        self.q1_opt.apply_gradients(zip(q1_grads, self.q1.trainable_weights))
        self.q2_opt.apply_gradients(zip(q2_grads, self.q2.trainable_weights))
        self.policy_opt.apply_gradients(zip(policy_grads, self.policy.trainable_weights))
        self.entropy_opt.apply_gradients([(entropy_grad, self.entropy_weight)])

        self.update_targ_network(self.q1_targ, self.q1.get_weights())
        self.update_targ_network(self.q2_targ, self.q2.get_weights())

        all_td_errors = tf.concat([td_error1, td_error2], axis = -1)
        mean_td_error = tf.reduce_mean(all_td_errors)
        
        return all_td_errors, (q1_loss, q2_loss, policy_loss, mean_entropy_loss, mean_td_error), ('Q1 loss', 'Q2 loss', 'Policy loss', 'Entropy loss', 'TD error')

        
class SequentialDuelingDiscreteSAC(RL_Algo):

    def __init__(self, 
                policy_architecture,
                q_architecture, 
                state_shape,
                num_actions,
                optimizer = tf.keras.optimizers.Adam,
                learning_rate = 0.0003,
                soft_update_beta = 0.995,
                discount = 0.997,
                initial_entropy = 1.0):
        # init parameters
        input_shape = (None, *state_shape)
        
        self.policy = policy_architecture
        self.policy.build(input_shape)
        self.policy_opt = optimizer(learning_rate)

        self.q1, self.q2, self.q1_targ, self.q2_targ = [tf.keras.models.clone_model(q_architecture) for i in range(4)]

        #bulids the four q-nets
        self.q1.build(input_shape)
        self.q1_targ.build(input_shape)
        self.q1_targ.set_weights(self.q1.get_weights())
        self.q1_opt = optimizer(learning_rate)

        self.q2.build(input_shape)
        self.q2_targ.build(input_shape)
        self.q2_targ.set_weights(self.q2.get_weights())
        self.q2_opt = optimizer(learning_rate)

        self.entropy_target = -1 * num_actions

        self.discount = discount
        self.soft_update_beta = soft_update_beta
        self.entropy_weight = tf.Variable(initial_entropy, trainable = True)
        self.entropy_opt = optimizer(learning_rate)

    def update_targ_network(self, targ_network, new_weights):
        targ_network.set_weights([
            self.soft_update_beta * targ_weight + (1.0 - self.soft_update_beta) * update_weights
            for targ_weight, update_weights in zip(targ_network.get_weights(), new_weights)
        ])

    def q_loss(self, q_val, action, target):
        #MSE, average across batch dimension
        td_error = tf.gather(q_val, action, axis = -1, batch_dims = 1) - target
        #print('TD error shape: ', td_error.get_shape())
        return tf.reduce_mean(tf.square(td_error)), tf.reduce_mean(td_error)

    def expectation_over_actions(self, action_distribution, q_val):
        expectation = tf.reduce_sum(tf.multiply(action_distribution, q_val - self.entropy_weight * tf.math.log(action_distribution)),axis = -1,keepdims = True)
        return expectation

    def update_step(self, s, a , r, s_next, done):

        q1_targs, q2_targs = self.q1_targ(s_next), self.q2_targ(s_next)

        min_q = tf.reduce_min(tf.stack([q1_targs, q2_targs]), axis = 0)

        next_state_distribution = self.policy(s_next)

        v_estimate = self.expectation_over_actions(next_state_distribution, min_q)

        q_target = r + self.discount * (1. - tf.dtypes.cast(done, 'float32')) * v_estimate

        with tf.GradientTape(persistent = True) as tape:

            action_distribution = self.policy(s)

            with tf.GradientTape(watch_accessed_variables=False) as entropy_tape:
                entropy_tape.watch(self.entropy_weight)
                entropy_loss = tf.reduce_sum(
                    tf.multiply(action_distribution, -self.entropy_weight * (tf.math.log(action_distribution) + self.entropy_weight)), 
                    -1, keepdims = True)
                mean_entropy_loss = tf.reduce_mean(entropy_loss)
            
            q1_vals, q2_vals = self.q1(s), self.q2(s)
            #use targets to compute losses
            q1_loss, td_error1 = self.q_loss(q1_vals, a, q_target)
            q2_loss, td_error2 = self.q_loss(q2_vals, a, q_target)
            #want to minimize the negative of the maximization objective
            policy_loss = -1 * tf.reduce_mean(self.expectation_over_actions(action_distribution, q1_vals))

        q1_grads, q2_grads = tape.gradient(q1_loss, self.q1.trainable_weights), tape.gradient(q2_loss, self.q2.trainable_weights)
        policy_grads = tape.gradient(policy_loss, self.policy.trainable_weights)
        entropy_grad = tape.gradient(mean_entropy_loss, self.entropy_weight)

        del tape

        self.q1_opt.apply_gradients(zip(q1_grads, self.q1.trainable_weights))
        self.q2_opt.apply_gradients(zip(q2_grads, self.q2.trainable_weights))
        self.policy_opt.apply_gradients(zip(policy_grads, self.policy.trainable_weights))
        self.entropy_opt.apply_gradients([(entropy_grad, self.entropy_weight)])

        self.update_targ_network(self.q1_targ, self.q1.get_weights())
        self.update_targ_network(self.q2_targ, self.q2.get_weights())
        
        all_td_errors = tf.concat([td_error1, td_error2], axis = -1)
        mean_td_error = tf.reduce_mean(all_td_errors)
        
        return all_td_errors, (q1_loss, q2_loss, policy_loss, mean_entropy_loss, mean_td_error), ('Q1 loss', 'Q2 loss', 'Policy loss', 'Entropy loss', 'TD error')

        


















        