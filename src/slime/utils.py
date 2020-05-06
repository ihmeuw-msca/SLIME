# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~
"""
import numpy as np


def sizes_to_indices(sizes):
    """Converting sizes to corresponding indices.
    Args:
        sizes (numpy.dnarray):
            An array consist of non-negative number.
    Returns:
        list{range}:
            List the indices.
    """
    u_id = np.cumsum(sizes)
    l_id = np.insert(u_id[:-1], 0, 0)

    return [
        np.arange(l, u) for l, u in zip(l_id, u_id)
    ]

def create_dummy_bounds():
    return np.array([-np.inf, np.inf])


def create_dummy_gprior():
    return np.array([0.0, np.inf])


def split_vec(x, sizes):
    assert len(x) == sum(sizes)
    return np.split(x, np.cumsum(sizes[:-1]))


def list_dot(x, y):
    return np.hstack(list(map(np.dot, x, y)))


def empty_array():
    return np.array(list())
