import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class PredictorBase(object):
    def __init__(self,  networks: []):
        self._networks = networks

    def reset(self):
        pass

    def networks_predict(self):
        pass

class ActorCriticTimeSeriesPredictor(PredictorBase):
    def __init__(self, networks: [], sequence_length: int):
        super(ActorCriticTimeSeriesPredictor, self).__init__(networks)

        self.features = []

        self._sequence_length = sequence_length

    def reset(self):
        self.features.clear()

    def data_element_add(self, features_element: []):
        if len(self.features) == self._sequence_length:
            del self.features[0]
            
        self.features.append(features_element)
        print(self.features)

    def networks_predict(self):
        print(len(self.features))
        print(self._sequence_length)
        if len(self.features) < self._sequence_length:
            return None, None, None
        
        values = []
        probabilities = []
        actions = []

        for network in self._networks:
            features = tf.convert_to_tensor([self.features], dtype=tf.float32)
            network_values, network_probabilities = network.call(features)
            network_action = self._action_choose(probabilities=network_probabilities)
            
            values.append(network_values)
            probabilities.append(network_probabilities)
            actions.append(network_action)

        return values, probabilities, actions

    def _action_choose(self, probabilities: tf.Tensor):
        action_probabilities = tfp.distributions.Categorical(probs=probabilities)
        action = action_probabilities.sample()

        return action