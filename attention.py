from typing import Dict, Tuple

import keras.backend as K
import tensorflow as tf
from keras.layers import Layer


class Attention(Layer):
    """ Attention mechanism. """

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape: Tuple):
        """ Build attention mechanism. """
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], 1), initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], 1), initializer="zeros"
        )
        super(Attention, self).build(input_shape)

    def call(self, x: tf.Tensor):
        """Attention mechanism. Calculates which part
        of input is important."""
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        return (input_shape[0], input_shape[-1])

    def get_config(self) -> Dict:
        return super(Attention, self).get_config()
