# -*- coding: utf-8 -*-
"""
    model
    ~~~~~
"""
import numpy as np
import scipy.optimize as sciopt

from .model import MRModel


class Optimizer:
    """Optimizer object.
    """

    def __init__(self, model: MRModel):
        self.objective = model.objective
        self.gradient = model.gradient
        self.var_bounds = model.var_bounds
        self.var_size = sum(model.var_sizes)
        self.opt_result = None

    def optimize(self):
        NotImplementedError()


class SciOptLBFGSB(Optimizer):
    """Scipy optimization LBFGSB solver.
    """

    def __init__(self, model):
        super().__init__(model)

    def optimize(self, x0=None, options=None):
        if x0 is None:
            x0 = np.zeros(self.var_size)
        self.opt_result = sciopt.minimize(
            fun=self.objective,
            x0=x0,
            jac=self.gradient,
            method='L-BFGS-B',
            bounds=self.var_bounds,
            options=options
        )
