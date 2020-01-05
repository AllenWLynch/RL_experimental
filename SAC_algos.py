import tensorflow as tf
import numpy as np
from utils import RL_Algo

class DiscreteSACwithValueNet(RL_Algo):

    def __init__(self, 
                q_net,
                policy_net,
                value_net, 
                optimizer,
                learning_rate,  
                entrop_weight,
                discount,
                state_shape,
                soft_update_beta = 0.995):
        # init parameters
        
        input_shape = (None, *state_shape)
        self.policy = policy_net
        self.policy.build(input_shape)

        self.q1 = q_net
        self.q1.build(input_shape)

        self.q2 = tf.keras.models.clone_model(self.q1)
        self.q2.build(input_shape)
        
        self.value_net = value_net
        self.value_net.build(input_shape)

        self.value_target = tf.keras.models.clone_model(self.value_net)
        self.value_target.build(input_shape)
        self.value_target.set_weights(self.value_net.get_weights())

        (self.policy_opt, self.q1_opt, self.q2_opt, self.v_opt) = [optimizer(learning_rate) for i in range(4)]
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


class DuelingDiscreteSAC(RL_Algo):

    def __init__(self, 
                policy_architecture,
                q_architecture, 
                optimizer,  
                learning_rate,
                entrop_weight,
                discount,
                state_shape,
                soft_update_beta = 0.995):
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

        self.entrop_weight = entrop_weight
        self.discount = discount
        #self.num_actions = num_actions
        self.soft_update_beta = soft_update_beta
        #self.replay = Replay(replay_size)

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
        expectation = tf.reduce_sum(tf.multiply(action_distribution, q_val - self.entrop_weight * tf.math.log(action_distribution)),axis = -1,keepdims = True)
        return expectation

    def update_step(self, s, a , r, s_next, done):

        with tf.GradientTape(persistent = True) as tape:

            q1_targs, q2_targs = self.q1_targ(s_next), self.q2_targ(s_next)

            min_q = tf.reduce_min(tf.stack([q1_targs, q2_targs]), axis = 0)

            action_distribution = self.policy(s)

            v_estimate = self.expectation_over_actions(action_distribution, min_q)

            q_target = r + self.discount * (1. - tf.dtypes.cast(done, 'float32')) * v_estimate

            q1_vals, q2_vals = self.q1(s), self.q2(s)
            #use targets to compute losses
            q1_loss, td_error = self.q_loss(q1_vals, a, q_target)
            q2_loss, _ = self.q_loss(q2_vals, a, q_target)

            #want to minimize the negative of the maximization objective
            policy_loss = -1 * tf.reduce_mean(self.expectation_over_actions(action_distribution, q1_vals))

        q1_grads, q2_grads = tape.gradient(q1_loss, self.q1.trainable_weights), tape.gradient(q2_loss, self.q2.trainable_weights)
        policy_grads = tape.gradient(policy_loss, self.policy.trainable_weights)

        del tape

        self.q1_opt.apply_gradients(zip(q1_grads, self.q1.trainable_weights))
        self.q2_opt.apply_gradients(zip(q2_grads, self.q2.trainable_weights))
        self.policy_opt.apply_gradients(zip(policy_grads, self.policy.trainable_weights))

        #for t, v in zip(self.q1_targ.get_weights(), self.q1.get_weights()):
        #    print(t.shape, v.shape)

        self.update_targ_network(self.q1_targ, self.q1.get_weights())
        self.update_targ_network(self.q2_targ, self.q2.get_weights())
        
        return (q1_loss, q2_loss, policy_loss, td_error), ('Q1 loss', 'Q2 loss', 'Policy loss','TD error')

    def get_action(self, state):
        #print(state, self.policy(tf.expand_dims(state, 0)))
        #assert(False)
        distribution = (self.policy(tf.expand_dims(state, 0))[0]).numpy()
        return np.random.choice(len(distribution), p = distribution)

        



















        