import layers as custom_layers
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers as keras_layers
from tensorflow.keras import models

class ActorCriticBase(keras.Model):
    def __init__(self, action_count: int, model_layers: []):
        super(ActorCriticBase, self).__init__()

        self._action_count = tf.Variable(action_count)
        self._last_action = None
        self._model_layers = model_layers
        self._probabilities_layer = keras_layers.Dense(action_count, activation='softmax')
        self._values_layer = keras_layers.Dense(1, activation=None)

    def call(self, input_state):
        model_output = self._layers_process(input_state)
        
        values = self._values_layer(model_output)
        probabilities = self._probabilities_layer(model_output)

        return values, probabilities
    
    def _layers_process(self, input_state):
        print("no layers are running")
        return input_state


class ActorCriticTransformer(ActorCriticBase):
    def __init__(self, action_count: int, feature_count: int, sequence_length: int,
                attention_head_count: int, attention_dense_size: int, dense_size: int,
                feed_forward_dimention: int, dropout_rate: float):
        model_layers = self._layers_build(feature_count, sequence_length, attention_head_count,
                attention_dense_size, dense_size, feed_forward_dimention, dropout_rate)
        super(ActorCriticTransformer, self).__init__(action_count, model_layers)
    
    def _layers_build(self, feature_count: int, sequence_length: int, attention_head_count: int,
                attention_dense_size: int, dense_size: int, feed_forward_dimention: int, dropout_rate: float):
        input_shape = (sequence_length, feature_count)

        model_layers = {}

        model_layers['transformer_encoder_1'] = custom_layers.Transformer(input_shape, attention_head_count,
                attention_dense_size, feed_forward_dimention, dropout_rate)
        model_layers['transformer_encoder_2'] = custom_layers.Transformer(input_shape, attention_head_count,
                attention_dense_size, feed_forward_dimention, dropout_rate)
        model_layers['transformer_encoder_3'] = custom_layers.Transformer(input_shape, attention_head_count,
                attention_dense_size, feed_forward_dimention, dropout_rate)

        model_layers['global_average_pooling_1D'] = keras_layers.GlobalAveragePooling1D(
                data_format='channels_first')
        model_layers['dropout_1'] = keras_layers.Dropout(dropout_rate)
        model_layers['dense'] = keras_layers.Dense(dense_size, activation='relu')
        model_layers['dropout_2'] = keras_layers.Dropout(dropout_rate)

        return model_layers
    
    def _layers_process(self, input_state):
        x = input_state

        x = self._model_layers['transformer_encoder_1']((x, x, x))
        x = self._model_layers['transformer_encoder_2']((x, x, x))
        x = self._model_layers['transformer_encoder_3']((x, x, x))

        x = self._model_layers['global_average_pooling_1D'](x)
        x = self._model_layers['dropout_1'](x)
        x = self._model_layers['dense'](x)
        x = self._model_layers['dropout_2'](x)
        print("through the layers")

        return x