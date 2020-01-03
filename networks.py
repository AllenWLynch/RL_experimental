

import tensorflow as tf
import numpy as np

def SimpleDense(layers):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(layer, activation = 'relu')
        for layer in layers
    ])