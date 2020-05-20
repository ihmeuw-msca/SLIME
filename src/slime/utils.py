# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~
"""
from typing import List, Union
import numpy as np


def sizes_to_indices(sizes: Union[List, np.ndarray]) -> List[np.ndarray]:
    """Convert sizes to indicies.

    Args:
        sizes (Union[List, np.ndarray]): Sizes of the given variables.

    Returns:
        List[np.ndarray]: Corresponding indices of the variables.
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


def is_bounds(x):
    ok = len(x) == 2
    ok = ok and (x[0] <= x[1])
    return ok


def is_gprior(x):
    ok = len(x) == 2
    ok = ok and (x[1] > 0.0)
    return ok
