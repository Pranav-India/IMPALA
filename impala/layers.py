import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Time2Vector(layers.Layer):
    def __init__(self, input_shape: (), sequence_length: int):
        super(Time2Vector, self).__init__()

        self._build(sequence_length)
    
    def _build(self, sequence_length: int):
        self._weights_linear = self.add_weight(shape=(sequence_length,), initializer='uniform', trainable=True)
        self._bias_linear = self.add_weight(shape=(sequence_length,), initializer='uniform', trainable=True)
        
        self._weights_periodic = self.add_weight(shape=(sequence_length,), initializer='uniform', trainable=True)
        self._bias_periodic = self.add_weight(shape=(sequence_length,), initializer='uniform', trainable=True)
    
    def call(self, input_state):
        input_state = tf.math.reduce_mean(input_state[:, :, :4], axis=-1) # Convert (batch, seq_len, 5) to (batch, seq_len)

        time_linear = (self._weights_linear * input_state) + self._bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1)

        time_periodic = tf.math.sin(tf.multiply(input_state, self._weights_periodic) + self._bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)

        return tf.concat([time_linear, time_periodic], -1)


class SingleAttention(layers.Layer):
    def __init__(self, input_shape: (), attention_dense_size: int):
        super(SingleAttention, self).__init__()

        self._attention_dense_size = tf.constant(float(attention_dense_size))

        self._build(input_shape, attention_dense_size)
        
    def _build(self, input_shape: (), attention_dense_size: int):
        self._query = layers.Dense(attention_dense_size, input_shape=input_shape, kernel_initializer='glorot_uniform',
                bias_initializer='glorot_uniform')
        self._key = layers.Dense(attention_dense_size, input_shape=input_shape, kernel_initializer='glorot_uniform',
                bias_initializer='glorot_uniform')
        self._value = layers.Dense(attention_dense_size, input_shape=input_shape, kernel_initializer='glorot_uniform',
                bias_initializer='glorot_uniform')
    
    def call(self, inputs: ()):
        query = self._query(inputs[0])
        key = self._key(inputs[1])
        value = self._value(inputs[2])

        attention_weights = tf.matmul(query, key, transpose_b=True)
        attention_weights = tf.map_fn(lambda x: x / np.sqrt(self._attention_dense_size), attention_weights)
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        
        return tf.matmul(attention_weights, value)


class MultiAttention(layers.Layer):
    def __init__(self, input_shape: (), attention_head_count: int, attention_dense_size: int,):
        super(MultiAttention, self).__init__()

        self._attention_heads = []
        self._attention_head_count = tf.constant(attention_head_count)

        self._build(input_shape, attention_dense_size)

    def _build(self, input_shape: (), attention_dense_size: int):
        feature_count = input_shape[-1]

        for _ in range(int(self._attention_head_count)):
            self._attention_heads.append(SingleAttention(input_shape, attention_dense_size))
        
        self._linear = layers.Dense(feature_count, input_shape=input_shape, kernel_initializer='glorot_uniform',
                bias_initializer='glorot_uniform')
    
    def call(self, inputs: ()):
        attention = [self._attention_heads[i](inputs) for i in range(self._attention_head_count)]
        concat = tf.concat(attention, -1)

        return self._linear(concat)


class Transformer(layers.Layer):
    def __init__(self, input_shape: (), attention_head_count: int, attention_dense_size: int,
                feed_forward_dimention: int, dropout_rate: float):
        super(Transformer, self).__init__()
        
        self._build(input_shape, attention_head_count, attention_dense_size, feed_forward_dimention, dropout_rate)

    def _build(self, input_shape: (), attention_head_count: int, attention_dense_size: int,
                feed_forward_dimention: int, dropout_rate: float):
        feature_count = input_shape[-1]

        self._attention_multi = MultiAttention(input_shape, attention_head_count, attention_dense_size)
        self._attention_dropout = layers.Dropout(dropout_rate)
        self._attention_normalization = layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self._feed_forward_conv1D_1 = layers.Conv1D(input_shape=input_shape, filters=feed_forward_dimention,
                kernel_size=1, activation='relu')
        self._feed_forward_conv1D_2 = layers.Conv1D(input_shape=input_shape, filters=feature_count, kernel_size=1)
        self._feed_forward_dropout = layers.Dropout(dropout_rate)
        self._feed_forward_normalization = layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    def call(self, inputs: ()):
        attention = self._attention_multi(inputs)
        attention = self._attention_dropout(attention)
        attention = self._attention_normalization(inputs[0] + attention)

        feed_forward = self._feed_forward_conv1D_1(attention)
        feed_forward = self._feed_forward_conv1D_2(feed_forward)
        feed_forward = self._feed_forward_dropout(feed_forward)
        feed_forward = self._feed_forward_normalization(inputs[0] + feed_forward)

        return feed_forward