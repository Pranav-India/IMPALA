from datetime import datetime
from datetime import timedelta
from impala import helpers
from impala import models
from impala.predictors import ActorCriticTimeSeriesPredictor
import logging
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import distribute
from tensorflow import optimizers


EPOCH_COUNT = 5

LEARNING_RATE = 0.0003
CONVICTION_ACTION_COUNT = 2
CONVICTION_FEATURE_COUNT = 21
POSITION_ACTION_COUNT = 10
POSITION_FEATURE_COUNT = 7
SEQUENCE_LENGTH = 240
ATTENTION_HEAD_COUNT = 12
ATTENTION_DENSE_SIZE = 256
DENSE_SIZE = 256
FEED_FORWARD_DIMENTION = 256
DROPOUT_RATE = 0.1

START_DATE = '2018-11-01'
END_DATE = '2018-11-07'


class Trainer(object):
    def __init__(self):
        pass

    def run(self):
        strategy = distribute.MirroredStrategy()

        # The agent is made up of two networks that rely on each other, and thus must be trained together. The
        # 'conviction' network does most of the heavy lifting for evaluating whether or not the agent should go long
        # or short on a stock. It has two actions: 0 for long and 1 for short.
        # 
        # The 'position' network is in charge of deciding how much should be bought or sold. The action that's output by
        # the conviction network is added as a feature for the position network, along with the probability for that
        # action and the current position in this stock held by the agent. The position network's actions are order
        # size multipliers that correspond to the following amounts: [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, float('inf')].
        # The base value that these is $10,000, meaning action [1] would result in trade in the size of: 0.1 * $10,000 = $1,000.
        # The final action, float(inf), represents liquidating the entire position held by the agent.
        with strategy.scope():
            self._conviction_network = models.ActorCriticTransformer(CONVICTION_ACTION_COUNT, CONVICTION_FEATURE_COUNT,
                    SEQUENCE_LENGTH, ATTENTION_HEAD_COUNT, ATTENTION_DENSE_SIZE, DENSE_SIZE, FEED_FORWARD_DIMENTION,
                    DROPOUT_RATE)
            self._conviction_network.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE))

            self._position_network = models.ActorCriticTransformer(POSITION_ACTION_COUNT, POSITION_FEATURE_COUNT,
                    SEQUENCE_LENGTH, ATTENTION_HEAD_COUNT, ATTENTION_DENSE_SIZE, DENSE_SIZE, FEED_FORWARD_DIMENTION,
                    DROPOUT_RATE)
            self._position_network.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE))
        
        combined_data = self._combined_data_get_as_dataset()

        for i in range(EPOCH_COUNT):
            rewards = strategy.run(self._day_process, (combined_data,))
            total_reward = strategy.reduce(distribute.ReduceOp.SUM, rewards, axis=None)

            logging.info('Epoch ' + str(i + 1) + ' finished with a total reward of ' + str(total_reward) + '!')
        
    def _combined_data_get_as_dataset(self):
        start_datetime = helpers.date_convert_to_datetime(START_DATE)
        end_datetime = helpers.date_convert_to_datetime(END_DATE)
        current_datetime = start_datetime - timedelta(days=1)

        combined_data = []

        while current_datetime < end_datetime:
            current_datetime += timedelta(days=1)
            current_date = helpers.datetime_convert_to_date(current_datetime)

            conviction_data = self._data_get_as_list('features', current_date)
            if conviction_data == None:
                continue

            position_data = self._data_get_as_list('position', current_date)
            evaluation_data = self._data_get_as_list('evaluation', current_date)

            combined_data.append((conviction_data, position_data, evaluation_data))

        # TODO: convert combined_data into into a tensor or tf.data.Dataset that's passable by the strategy
        
        return combined_data

    # This is a stand-in function for this example project. Data is retreived from Cloud Storage for actual training. 
    def _data_get_as_list(self, data_type: str, date: str):
        path = os.path.join(os.path.dirname(__file__), 'data', data_type, date)
        
        if os.path.exists(path) == False:
            return None

        data = []
        with open(path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line_elements = line.split('\t')
                element_data = []

                for element in line_elements:
                    element_data.append(float(element))
                
                data.append(element_data)
        
        return data
    
    # All of the data is broken up into individual days, which is treated as a single 'episode' of training.
    # The strategy should pass every process one day's worth of data at a time. A single day should be evaluated
    # all at once.
    def _day_process(self, combined_data):
        # TODO: extract conviction_data, position_data, and evaluation_data from combined_data
        conviction_data = []
        position_data = []
        evaluation_data = []

        conviction_predictor = ActorCriticTimeSeriesPredictor([self._conviction_network], SEQUENCE_LENGTH)
        position_predictor = ActorCriticTimeSeriesPredictor([self._position_network], SEQUENCE_LENGTH)
        conviction_predictions = []
        position_predictions = []
        rewards = []

        # The values, probabilities, actions, and rewards for each step through this training episode are pre-calculated
        # and saved so that the new state 'value' can be reused when calling the learn function.
        for i in range(len(conviction_data)):
            values, probabilities, actions = conviction_predictor.networks_predict()
            conviction_predictor.data_element_add(conviction_data[i])
            position_data_element = position_data[i]
            
            # values will be returned as None if the number of feature elements stored in the conviction_predictor is
            # below the SEQUENCE_LENGTH threshold.
            if values == None:
                # The position_data elements do not have all of the features required by the position_network.
                # Missing features need to be added during evaluation. This is done here in the event that the stored
                # feature elements is below SEQUENCE_LENGTH, otherwise it's done below.
                position_data_element.extend([random.randint(0, 1), 0.5, 0])
                position_predictor.data_element_add(position_data_element)
                continue
            
            # ActorCriticTimeSeriesPredictor returns a list of values, probabilities, and actions. This is to allow
            # for multiple models to be evaluated using the same data. This functionality is not necessary during
            # training, but that's why we need to get the 0th element here and below.
            conviction_predictions.append(
            {
                'value': values[0],
                'probabilities': probabilities[0],
                'action': actions[0]
            })

            conviction_action_index = int(actions[0][0])
            bid_close_price = float(position_data_element[2])
            ask_close_price = float(position_data_element[3])
            # This value assignment is a stand-in for a more involved calculation that I haven't included in order to
            # streamline this example.
            position_relative_size, position_average_price = 1.0, 1.0
            
            # Changes the bid/ask values to be relative to the average price of the agent's current position in the stock.
            position_data_element[2] = (bid_close_price / position_average_price) - 1
            position_data_element[3] = (ask_close_price / position_average_price) - 1
            position_data_element.append(conviction_action_index)
            position_data_element.append(probabilities[0][conviction_action_index])
            position_data_element.append(position_relative_size)
            position_predictor.data_element_add(position_data_element)

            values, probabilities, actions = position_predictor.networks_predict()
            position_predictions.append(
            {
                'value': values[0],
                'probabilities': probabilities[0],
                'action': actions[0]
            })

            current_price = float(evaluation_data[i][0])
            # This append is a stand-in for a more involved calculation that I haven't included in order to
            # streamline this example. current_price is used here to calculate this value.
            rewards.append(random.randint(-1, 1))
        
        for i in range(len(conviction_predictions) - 1):
            is_done = i == len(conviction_predictions) - 2
            
            # TODO

        # Important that the total reward is passed for logging purposes.
        return tf.constant(np.sum(rewards))