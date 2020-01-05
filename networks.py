

import tensorflow as tf
import numpy as np

def SimpleDense(layers):
    model = tf.keras.Sequential()
    for layer in layers:
        model.add(tf.keras.layers.Dense(layer))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
    return model


def SimpleDueling(layers, input_shape, num_actions):

    state = tf.keras.Input(shape = input_shape)

    features = SimpleDense(layers)(state)

    value = tf.keras.layers.Dense(1, activation = 'linear')(features)

    advantages = tf.keras.layers.Dense(num_actions, activation = 'linear')(features)

    Q_vals = DuelingQ()((value, advantages))

    return tf.keras.Model(state, Q_vals)


class DuelingQ(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shapes):
        assert(len(input_shapes) == 2), 'Expected two inputs: the value and action stream'        

    def call(self, inputs):

        (state_value, advantage) = inputs

        return state_value + (advantage - tf.reduce_mean(advantage))

def ConvLNRelu(channels, filter_size, strides):
    return tf.keras.Sequential([
        tf.keras.layers.Conv3D(channels, filter_size, strides = strides, padding = 'same'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.ReLU(),
    ])

class ImageTransformerLayer(tf.keras.layers.Layer):

    def __init__(self, k = 8, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        (_, depth, h, w, nc) = input_shape
        reduced_channels = nc // self.k
        num_features = h*w
        self.q_conv = tf.keras.layers.Conv3D(reduced_channels, 1)
        self.k_conv = tf.keras.layers.Conv3D(reduced_channels, 1)
        self.v_conv = tf.keras.layers.Conv3D(nc, 1)
        self.mlp_conv = tf.keras.layers.Conv3D(nc, 1, activation = 'relu')
        self.feature_flattener = tf.keras.layers.Reshape((depth, num_features, -1))
        self.softmaxer = tf.keras.layers.Softmax(axis = -1)
        self.unflattener = tf.keras.layers.Reshape((depth, h, w, nc))
        self.alpha_shaper = tf.keras.layers.Reshape((depth, num_features, depth, num_features))
        self.gamma = tf.Variable(0., 'float32')
        self.attn_layer_norm = tf.keras.layers.LayerNormalization()
        self.mlp_layer_norm = tf.keras.layers.LayerNormalization()
    
    def call(self, X):
        #shape (m, d, f, c)
        Q = self.feature_flattener(self.q_conv(X))
        K = self.feature_flattener(self.k_conv(X))
        V = self.feature_flattener(self.v_conv(X))

        KT = tf.transpose(K, perm = (0,1,3,2))
        #shape (m, d, f, d2, f2)
        energies = tf.einsum('bijk,blkm->bijlm', Q, KT)

        #flatten and softmax on last dimension
        #shape (m, d, f, d2*f2)
        energies = self.feature_flattener(energies)

        alphas = self.softmaxer(energies)

        alphas = self.alpha_shaper(alphas)

        #alphas (m, d, f, d, f)
        #V (m, d, f, c)
        attn = tf.einsum('bdfgh,bdhi->bdfi',alphas, V)

        resized = self.unflattener(attn)

        attn_layer_ouput = self.attn_layer_norm(resized + X)

        output = self.mlp_layer_norm(attn_layer_ouput + self.mlp_conv(attn_layer_ouput))

        return output

def frame_Q_Transformer(input_shape, num_actions):

    #(depth, h, w, nc) = input_shape

    frames = tf.keras.Input(shape = input_shape)

    #in:256
    X = ConvLNRelu(32, (1, 4,4), (1,2,2))(frames)

    #in: 128
    X = ConvLNRelu(64, (1, 4,4), (1,2,2))(X)

    #in: 64
    X = ConvLNRelu(128, (1, 4,4), (1,2,2))(X)

    #in: 32
    X = ConvLNRelu(256, (1, 4,4), (1,2,2))(X)

    #in: 16
    #advantage stream
    A = ImageTransformerLayer(k = 4)(X)

    A = ImageTransformerLayer(k = 4)(A)[:,-1]

    A = tf.keras.layers.Conv2D(16, 1)(A)
    A = tf.keras.layers.BatchNormalization()(A)
    A = tf.keras.layers.ReLU()(A)
    A = tf.keras.layers.Flatten()(A)

    A = tf.keras.layers.Dense(num_actions)(A)

    #value stream
    V = ImageTransformerLayer(k = 4)(X)

    V = ImageTransformerLayer(k = 4)(V)[:,-1]

    V = tf.keras.layers.Conv2D(16, 1)(V)
    V = tf.keras.layers.BatchNormalization()(V)
    V = tf.keras.layers.ReLU()(V)
    V = tf.keras.layers.Flatten()(V)

    V = tf.keras.layers.Dense(1)(V)

    Q_vals = DuelingQ()((V, A))

    return tf.keras.Model(frames, Q_vals)