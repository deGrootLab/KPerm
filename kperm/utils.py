"""helpers methods

"""
import time
from functools import wraps
import logging
import sys


def _write_list_of_tuples(filename, data):
    with open(filename, 'w', encoding="utf-8") as f:
        for d in data:
            line = ' '.join([str(x) for x in d]) + '\n'
            f.write(line)


def _count_time(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end = time.time()
            print(f"Total execution time: {end - start:.5f} s\n\n")

    return _time_it


def _create_logger(loc):
    # remove existing handlers, if any
    logger = logging.getLogger("kchannel")
    logger.handlers = []

    logger = logging.getLogger("kchannel")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(loc, mode="w")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger
