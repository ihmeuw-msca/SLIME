# -*- coding: utf-8 -*-
"""
    cov_model
    ~~~~~~~~~
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np
from .data import StaticData, MRData
from .utils import create_dummy_bounds, create_dummy_gprior
from .utils import split_vec, list_dot, is_bounds, is_gprior


@dataclass
class Covariate(StaticData):
    """Covariate model settings
    """
    name: str
    use_re: bool = False
    bounds: np.ndarray = field(default_factory=create_dummy_bounds)
    gprior: np.ndarray = field(default_factory=create_dummy_gprior)
    re_var: float = np.inf

    def __post_init__(self):
        super().__post_init__()
        assert is_bounds(self.bounds)
        assert is_gprior(self.gprior)
        assert self.re_var > 0.0

    def get_var_size(self, data: MRData) -> int:
        """Get optimization variable size.
        """
        return data.num_groups if self.use_re else 1

    def get_cov_data(self, data: MRData) -> List[np.ndarray]:
        """Get the covariate.
        """
        assert self.name in data.covs
        cov = data.covs[self.name]
        if self.use_re:
            return split_vec(cov, data.group_sizes)
        else:
            return [cov]

    def get_var_bounds(self, data: MRData) -> np.ndarray:
        """Get optimization variable bounds as the 2D array.
        """
        return np.repeat(self.bounds[None, :], self.get_var_size(data), axis=0)


class MRModel:
    """Covariate Model
    """
    def __init__(self,
                 covariates: List[Covariate],
                 data: MRData = None):
        """Constructor of the covariate model.
        """
        self.covariates = covariates

        self.obs = None
        self.obs_se = None
        self.cov_data = None
        self.group_sizes = None
        self.var_sizes = None
        self.var_bounds = None

        if data is not None:
            self.attach_data(data)

    def attach_data(self, data: MRData):
        """Attach the data.
        """
        self.obs = data.obs
        self.obs_se = data.obs_se
        self.cov_data = [
            cov.get_cov_data(data)
            for cov in self.covariates
        ]
        self.group_sizes = data.group_sizes
        self.var_sizes = np.array([
            cov.get_var_size(data)
            for cov in self.covariates
        ])
        self.var_bounds = np.vstack([
            cov.get_var_bounds(data)
            for cov in self.covariates
        ])

    def detach_data(self):
        """Detach the object from the data.
        """
        self.obs = None
        self.obs_se = None
        self.cov_data = None
        self.group_sizes = None
        self.var_sizes = None
        self.var_bounds = None

    def objective(self, x: np.ndarray) -> float:
        """Optimization objective function.
        """
        # convert optimization variable to total effect for cov and group
        effects = split_vec(x, self.var_sizes)
        # compute the prediction
        prediction = sum([
            list_dot(effect, self.cov_data[i])
            for i, effect in enumerate(effects)
        ])
        # compute residual
        residual = self.obs - prediction
        # compute the negative likelihood from the data part
        val = 0.5*np.sum((residual/self.obs_se)**2)
        # compute the Gaussian prior
        val += sum([
            0.5*np.sum(((effects[i] - cov.gprior[0])/cov.gprior[1])**2)
            for i, cov in enumerate(self.covariates)
        ])
        # compute the random effects prior
        val += sum([
            0.5*np.sum((effects[i] - np.mean(effects[i]))**2)/cov.re_var
            for i, cov in enumerate(self.covariates)
        ])
        return val

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Optimization gradient function.
        """
        # convert optimization variable to total effect for cov and group
        effects = split_vec(x, self.var_sizes)
        # compute the prediction
        prediction = sum([
            list_dot(effect, self.cov_data[i])
            for i, effect in enumerate(effects)
        ])
        # compute scaled residual
        residual = (self.obs - prediction)/self.obs_se**2
        s_residual = split_vec(residual, self.group_sizes)
        # gradient from the data likelihood
        grad = np.hstack([
            list_dot(self.cov_data[i], s_residual) if cov.use_re else
            list_dot(self.cov_data[i], [residual])
            for i, cov in enumerate(self.covariates)
        ])
        # gradient from the Gaussian prior
        grad += np.hstack([
            (effects[i] - cov.gprior[0])/cov.gprior[1]**2
            for i, cov in enumerate(self.covariates)
        ])
        # gradient from the random effects prior
        grad += np.hstack([
            (effects[i] - np.mean(effects[i]))/cov.re_var
            for i, cov in enumerate(self.covariates)
        ])
        return grad
