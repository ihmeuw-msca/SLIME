# -*- coding: utf-8 -*-
"""
    cov_model
    ~~~~~~~~~
"""
import numpy as np
from slime.core import MRData
import slime.core.utils as utils

class CovModel:
    """Single covariate model.
    """
    def __init__(self, col_cov,
                 use_re=False,
                 bounds=None,
                 gprior=None,
                 re_var=1.0):
        """Constructor CovModel.

        Args:
            col_cov (str): Column for the covariate.
            use_re (bool, optional): If use the random effects.
            bounds (np.ndarray | None, optional):
                Bounds for the covariate multiplier.
            gprior (np.ndarray | None, optional):
                Gaussian prior for the covariate multiplier.
            re_var (float, optional):
                Variance of the random effect, if use random effect.
        """
        self.col_cov = col_cov
        self.use_re = use_re
        self.bounds = bounds
        self.gprior = gprior
        self.re_var = re_var

        self.name = self.col_cov
        self.var_size = None
        self.cov = None
        self.group_sizes = None

    def attach_data(self, data):
        """Attach the data.

        Args:
            data (MRData): MRData object.
        """
        assert self.col_cov in data.df
        if self.use_re:
            self.var_size = data.num_groups
        else:
            self.var_size = 1
        self.cov = data.df[self.col_cov].values
        self.group_sizes = data.group_sizes

    def detach_data(self):
        """Detach the object from the data.
        """
        self.var_size = None
        self.cov = None
        self.group_sizes = None

    def get_cov_multiplier(self, x):
        """Transform the effect to the optimization variable.

        Args:
            x (np.ndarray): optimization variable.
        """
        if self.use_re:
            return np.repeat(x, self.group_sizes)
        else:
            return np.repeat(x, self.group_sizes.sum())

    def predict(self, x):
        """Predict for the optimization problem.

        Args:
            x (np.ndarray): optimization variable.
        """
        cov_multiplier = self.get_cov_multiplier(x)
        return self.cov*cov_multiplier

    def prior_objective(self, x):
        """Objective related to prior.

        Args:
            x (np.ndarray): optimization variable.
        """
        val = 0.0

        # random effects priors
        if self.use_re:
            val += 0.5*np.sum((x - np.mean(x))**2)/self.re_var

        # Gaussian prior for the effects
        if self.gprior is not None:
            val += 0.5*np.sum(((x - self.gprior[0])/self.gprior[1])**2)

        return val

    def extract_bounds(self):
        """Extract the bounds for the optimization problem.
        """
        if self.bounds is None:
            bounds = np.array([-np.inf, np.inf])
        else:
            bounds = self.bounds

        return np.repeat(bounds[None, :], self.var_size, axis=0)


class CovModelSet:
    """A set of CovModel.
    """
    def __init__(self, cov_models, data=None):
        """Constructor of the covariate model set.

        Args:
            cov_models (list{CovModel}): A list of covaraite set.
            data (MRData | None, optional): Data to be attached.
        """
        assert isinstance(cov_models, list)
        assert all([isinstance(cov_model, CovModel)
                    for cov_model in cov_models])
        self.cov_models = cov_models

        self.var_size = None
        self.var_sizes = None
        self.var_idx = None

        if data is not None:
            self.attach_data(data)

    def attach_data(self, data):
        """Attach the data.

        Args:
            data (MRData): MRData object.
        """
        for cov_model in self.cov_models:
            cov_model.attach(data)

        self.var_sizes = np.array([
            cov_model.var_size for cov_model in self.cov_models
        ])
        self.var_size = np.sum(self.var_size)
        self.var_idx = utils.sizes_to_indices(self.var_sizes)

    def detach_data(self):
        """Detach the object from the data.
        """
        for cov_model in self.cov_models:
            cov_model.detach()

        self.var_size = None
        self.var_sizes = None
        self.var_idx = None

    def predict(self, x):
        """Predict for the optimization.

        Args:
            x (np.ndarray): optimization variable.
        """
        return np.sum([
            cov_model.predict(x[self.var_idx[i]])
            for i, cov_model in enumerate(self.cov_models)
        ], axis=0)

    def prior_objective(self, x):
        """Objective related to prior.

        Args:
            x (np.ndarray): optimization variable.
        """
        return np.sum([
            cov_model.prior_objective(x[self.var_idx[i]])
            for i, cov_model in enumerate(self.cov_models)
        ])

    def extract_bounds(self):
        """Extract the bounds for the optimization problem.
        """
        return np.vstack([
            cov_model.extract_bounds()
            for cov_model in self.cov_models
        ])
