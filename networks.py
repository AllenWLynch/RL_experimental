

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

class TransformerMaskGenerator(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):

        print('here:', input_shape)

        time_steps = input_shape[0][1]

        self.lookahead_mask = tf.linalg.band_part(tf.ones((time_steps,time_steps)), -1, 0)
        self.lookbehind_mask = tf.linalg.band_part(tf.ones((time_steps, time_steps)), time_steps//2 - 1, -1)

        self.combined_mask = self.lookahead_mask * self.lookbehind_mask

    def call(self, X):
        (state_mask, burn_in_mask) = X
        #state_mask (m, d)
        #burn_in_mask (d,)

        state_square = state_mask[:,:,tf.newaxis] * state_mask[:, tf.newaxis, :]

        softmax_mask =  state_square * tf.expand_dims(self.combined_mask,0)

        return softmax_mask, state_mask * burn_in_mask


class ImageTransformerLayer(tf.keras.layers.Layer):

    def __init__(self, positional_encoder, k = 1, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.pe = positional_encoder

    def build(self, input_shape):
        (_, depth, h, w, nc) = input_shape

        reduced_channels = nc // self.k
        num_features = h*w
        self.q_conv = tf.keras.layers.Conv3D(reduced_channels, 1)
        self.k_conv = tf.keras.layers.Conv3D(reduced_channels, 1)
        self.v_conv = tf.keras.layers.Conv3D(nc, 1)
        self.mlp_conv = tf.keras.layers.Conv3D(nc, 1, activation = 'relu')
        self.kq_flattener = tf.keras.layers.Reshape((depth, num_features, reduced_channels))
        self.v_flattener = tf.keras.layers.Reshape((depth, num_features, nc))
        self.softmaxer = tf.keras.layers.Softmax(axis = -1)
        self.unflattener = tf.keras.layers.Reshape((depth, h, w, nc))
        self.alpha_shaper = tf.keras.layers.Reshape((depth, num_features, depth, num_features))
        self.gamma = tf.Variable(0., 'float32')
        self.attn_layer_norm = tf.keras.layers.LayerNormalization()
        self.mlp_layer_norm = tf.keras.layers.LayerNormalization()
        self.r = reduced_channels

        '''if not self.pe.built:
            print('there')
            self.pe.build((None, depth, num_features, reduced_channels))'''
    
    def call(self, X, mask = None):
        #shape (m, d, f, c)
        Q = self.kq_flattener(self.q_conv(X))
        K = self.kq_flattener(self.k_conv(X))
        V = self.v_flattener(self.v_conv(X))

        KT = tf.transpose(K, perm = (0,1,3,2))
        #shape (m, d, f, d2, f2)
        positional_embeds = self.pe(Q)

        energies = (tf.einsum('bijk,blkm->bijlm', Q, KT) + positional_embeds)/tf.sqrt(tf.dtypes.cast(self.r, tf.float32))
        
        #add mask to non-states
        if not mask is None:
            #mask (m, d, d) => (m, d, 1, d, 1)
            softmax_mask = mask[:,:,tf.newaxis,:,tf.newaxis]
            #invert mask, set masked vals to -infinity for softmax to zero
            energies = energies + (1. - softmax_mask) * -1e6
        
        #flatten and softmax on last dimension
        #shape (m, d, f, d2*f2)
        energies = self.v_flattener(energies)

        alphas = self.softmaxer(energies)

        alphas = self.alpha_shaper(alphas)

        #alphas (m, d, f, d, f)
        #V (m, d, f, c)
        attn = tf.einsum('bdfgh,bdhi->bdfi',alphas, V)

        resized = self.unflattener(attn)

        attn_layer_ouput = self.attn_layer_norm(resized + X)

        output = self.mlp_layer_norm(attn_layer_ouput + self.mlp_conv(attn_layer_ouput))

        return output


class RelativePositionalEncoder(tf.keras.layers.Layer):

    def __init__(self, max_relative_distance, **kwargs):
        super().__init__(**kwargs)
        self.max_relative_distance = max_relative_distance
        self.clipping_fn = lambda x,s : tf.maximum(-s, tf.minimum(s, x))

    def build(self, input_shape):
        
        assert(len(input_shape)) == 4, 'Input to positional encoder must be a attention-matrix (Q,K,or V) with shape: (m, d, f, c)'
        (_, d, f, c) = input_shape

        #print(input_shape)

        embeddings_shape = (2 * (d - 1) + 1, f * c)
        #print(embeddings_shape)
        self.pe = self.add_weight(shape = embeddings_shape, name = 'positional_embeddings')
        self.q_flattener = tf.keras.layers.Reshape((d, f*c))

        d_ = np.arange(d)[:, np.newaxis]
        
        relative_distances = (d_ - d_.T)
        relative_distances = self.clipping_fn(relative_distances, self.max_relative_distance)
        relative_distances += min(self.max_relative_distance + 1, d) - 1

        self.relative_distances_mat = relative_distances
        #self.pe_matrix = tf.gather(self.pe, relative_distances)

    def call(self, attn_x):
        #(m, d, f, c) => (m, d, f*c)
        flat = self.q_flattener(attn_x)
        
        #(m, d1, f*c) dot (d1, f*c, d2) => (m, d1, d2)
        embeddings = tf.gather(self.pe, self.relative_distances_mat)

        embeddings = tf.expand_dims(tf.transpose(embeddings, perm = [0,2,1]),0)

        x = tf.einsum('bdq,bdqi->bdi', flat, embeddings)
        #(m, d1, d2) => (m, d1, 1, d2, 1)
        output = x[:,:,tf.newaxis,:,tf.newaxis]

        return output

def ImageTransformer(input_shape, max_embedding_distance, num_layers):

    (d, h, w, nc) = input_shape

    X = tf.keras.Input(shape = input_shape, name = 'image_sequence')

    state_mask = tf.keras.Input(shape = (d,), name = 'state_mask')

    burn_in_mask = tf.keras.Input(shape = (d,), name = 'burn-in_mask')

    positional_embedder = RelativePositionalEncoder(max_embedding_distance)

    softmax_mask, loss_mask = TransformerMaskGenerator()((state_mask, burn_in_mask))

    attn = ImageTransformerLayer(positional_embedder)(X, mask = softmax_mask)

    for i in range(num_layers - 1):
        attn = ImageTransformerLayer(positional_embedder)(attn, mask = softmax_mask)

    return tf.keras.Model(inputs = [X, state_mask, burn_in_mask], outputs = [attn, loss_mask])

def DiscreteQ_Transformer(input_shape, filter_sizes, max_embedding_distance):
    #(depth, h, w, nc) = input_shape

    assert(len(filter_sizes) == 4), 'Must provide four filter sizes as a tuple ex: (4, 4, 3, 3)'

    frames = tf.keras.Input(shape = input_shape)

    #in:256
    X = ConvLNRelu(32, (1, filter_sizes[0], filter_sizes[0]), (1,2,2))(frames)

    #in: 128
    X = ConvLNRelu(64, (1, filter_sizes[1], filter_sizes[1]), (1,2,2))(X)

    #in: 64
    X = ConvLNRelu(128, (1, filter_sizes[2], filter_sizes[2]), (1,2,2))(X)

    #in: 32
    X = ConvLNRelu(256, (1, filter_sizes[3], filter_sizes[3]), (1,2,2))(X)

    #in: 16
    #advantage stream
    A, loss_mask = ImageTransformer(X.get_shape(), max_embedding_distance, 3)
    A = tf.keras.layers.Flatten()(A)
    A = tf.keras.layers.Dense(num_actions)(A)

    #value stream
    V, _ = ImageTransformer(X.get_shape(), max_embedding_distance, 3)
    V = tf.keras.layers.Flatten()(V)
    V = tf.keras.layers.Dense(1)(V)

    Q_vals = DuelingQ()((V, A))

    return tf.keras.Model(frames, [Q_vals, loss_mask])