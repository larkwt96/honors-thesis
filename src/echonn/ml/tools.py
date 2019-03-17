import operator
from functools import reduce
import numpy as np


def product(args):
    return reduce(operator.mul, args, 1)


def get_lengths(shapes):
    """ get vector of lengths (number of elements) of weight matrices """
    return [product(shape) for shape in shapes]


def unpack(w, shapes, lengths=None):
    ''' given a weights vector, build w parts from it '''
    if lengths is None:
        lengths = get_lengths(shapes)
    last_length = 0
    ws = []
    for shape, length in zip(shapes, lengths):
        next_length = last_length + length
        ws.append(w[last_length:next_length].reshape(shape))
        last_length = next_length
    return ws


def init_weights(shapes):
    ''' generate weights vector '''
    lengths = get_lengths(shapes)
    w = np.random.uniform(-1, 1, size=sum(lengths))
    return w, unpack(w, shapes, lengths)
