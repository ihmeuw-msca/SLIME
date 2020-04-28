# -*- coding: utf-8 -*-
"""
    model
    ~~~~~
"""
import numpy as np
import scipy.optimize as sciopt
from slime.core import MRData
from .cov_model import CovModelSet


class MRModel:
    """Linear MetaRegression Model.
    """
    def __init__(self, data, cov_models):
        """Constructor of the MetaRegression Model.

        Args:
            data (MRData): Data object
            cov_models (CovModelSet): Covariate models.
        """
        self.data = data
        self.cov_models = cov_models

        # unpack data
        self.obs = data.df[data.col_obs].values
        self.obs_se = data.df[data.col_obs_se].values

        # pass data into the covariate models
        self.cov_models.attach_data(self.data)
        self.bounds = self.cov_models.extract_bounds()
        self.opt_result = None
        self.result = None

    def objective(self, x):
        """Objective function for the optimization.

        Args:
            x (np.ndarray): optimization variable.
        """
        # data
        prediction = self.cov_models.predict(x)
        val = 0.5*np.sum(((self.obs - prediction)/self.obs_se)**2)

        # prior
        val += self.cov_models.prior_objective(x)

        return val

    def gradient(self, x):
        """Gradient function for the optimization.

        Args:
            x (np.ndarray): optimization variable.
        """
        finfo = np.finfo(float)
        step = finfo.tiny/finfo.eps
        x_c = x + 0j
        grad = np.zeros(x.size)
        for i in range(x.size):
            x_c[i] += step*1j
            grad[i] = self.objective(x_c).imag/step
            x_c[i] -= step*1j

        return grad

    def fit_model(self, x0=None, options=None):
        """Fit the model, including initial condition and parameter.
        Args:
            x0 (np.ndarray, optional):
                Initial guess for the optimization variable.
            options (None | dict):
                Optimization solver options.
        """
        if x0 is None:
            x0 = np.zeros(self.cov_models.var_size)
        self.opt_result = sciopt.minimize(
            fun=self.objective,
            x0=x0,
            jac=self.gradient,
            method='L-BFGS-B',
            bounds=self.bounds,
            options=options
        )

        self.result = self.cov_models.process_result(self.opt_result.x)
