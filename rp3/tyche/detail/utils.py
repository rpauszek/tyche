import numpy as np
from functools import wraps

def row(x):
    return np.array(x)[np.newaxis, :]


def col(x):
    return np.array(x)[:, np.newaxis]


def numpy_preprocessor(cls):
    @wraps(cls)
    def wrapper(*args):
        new_args = [np.array(arg) for arg in args]
        return cls(*new_args)
    return wrapper
