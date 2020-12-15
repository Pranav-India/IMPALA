from datetime import datetime
import logging
import pytz
import sys


def date_convert_to_datetime(date: str):
    return pytz.timezone('America/New_York').localize(datetime.strptime(date, '%Y-%m-%d'))

def _datetime_digit_format(digit: int):
    digit = str(digit)
    if len(digit) == 1:
        digit = "0" + digit

    return digit

def datetime_convert_to_date(input_datetime: datetime):
    month = _datetime_digit_format(input_datetime.month)
    day = _datetime_digit_format(input_datetime.day)

    return str(input_datetime.year) + '-' + month + '-' + day

def logging_setup():
    root_logger = logging.getLogger()
    root_logger_previous_handlers = list(root_logger.handlers)
    for h in root_logger_previous_handlers:
        root_logger.removeHandler(h)
    root_logger.setLevel(logging.INFO)
    root_logger.propagate = False

    # Set tf logging to avoid duplicate logging. If the handlers are not removed then we will have duplicate
    # logging
    tf_logger = logging.getLogger('TensorFlow')
    while tf_logger.handlers:
        tf_logger.removeHandler(tf_logger.handlers[0])

    # Redirect INFO logs to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    root_logger.addHandler(stdout_handler)