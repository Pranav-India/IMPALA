
import numpy as np
import helpers

START_DATE = '2018-11-01'
END_DATE = '2018-11-07'

class Environment():
    def __init__(self , sequence_length):
        self.sequence_length = sequence_length
        
    def reset():
        data = self._combined_data_get_as_dataset()
        conviction_data = data[0][0]
        position_data = data[0][1]
        evaluation_data = data[0][2]

        
        return conviction_state 

    def step():
        return next_state , reward

    def _combined_data_get_as_dataset(self):
        start_datetime = helpers.date_convert_to_datetime(START_DATE)
        end_datetime = helpers.date_convert_to_datetime(END_DATE)
        current_datetime = start_datetime - timedelta(days=1)

        combined_data = []

        while current_datetime < end_datetime:
            current_datetime += timedelta(days=1)
            current_date = helpers.datetime_convert_to_date(current_datetime)

            conviction_data = self._data_get_as_list('conviction', current_date + str('.txt'))
            if conviction_data == None:
                continue

            position_data = self._data_get_as_list('position', current_date + str('.txt'))
            evaluation_data = self._data_get_as_list('evaluation', current_date + str('.txt'))

            combined_data.append((conviction_data, position_data, evaluation_data))

        return combined_data

    def _data_get_as_list(self, data_type: str, date: str):
        path = os.path.join(os.path.dirname(__file__), 'data', data_type, date)

        
        #print(path,"this is path")
        if os.path.exists(path) == False:
            #print("true")
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
        
        data = tf.convert_to_tensor(data)
        return data
