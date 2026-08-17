import itertools, logging, time, os
import colorlog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from time import time


def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def datetime_suffix():
    return datetime.now().strftime("%y%m%d%H%M%S")


def random_seed():
    t = 1000 * time.time()  # current time in milliseconds
    seed = int(t) % 2**32
    return seed


def set_home_folder_prefix(path_dict):
    nodename = os.uname()[1]
    if nodename in path_dict.keys():
        return path_dict[nodename]
    else:
        return path_dict["default"]


def create_logger(logger_name="this project", logging_level="INFO"):
    # level = "info", "warn"m "debug", "error"
    FORMAT = "\n%(log_color)s[%(asctime)s - %(name)s - {%(filename)s:%(lineno)d} - %(levelname)s] | %(message)s"
    DATEF = "%H:%M:%S"

    log_colors = {
        "DEBUG": "purple",
        "INFO": "green",
        "WARNING": "blue",
        "ERROR": "yellow",
        "CRITICAL": "bold_red",
    }

    formatter = colorlog.ColoredFormatter(FORMAT, datefmt=DATEF, log_colors=log_colors)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    mylogger = colorlog.getLogger(logger_name)
    mylogger.setLevel(logging_level)
    mylogger.addHandler(handler)

    return mylogger


glogger = create_logger("global", "INFO")


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()

        class_name = (
            args[0].__class__.__name__
            if args and hasattr(args[0], "__class__")
            else None
        )
        function_name = f"{class_name}.{func.__name__}" if class_name else func.__name__

        glogger.info(f"Function {function_name!r} executed in {(t2 - t1):.4f}s")
        return result

    return wrap_func
