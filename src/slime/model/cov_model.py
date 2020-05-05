# -*- coding: utf-8 -*-
"""
    cov_model
    ~~~~~~~~~
"""
from dataclasses import dataclass, field
from typing import Dict, List, Union
import numpy as np
from slime.core import MRData
import slime.core.utils as utils


@dataclass
class Covariate:
    """Covariate model settings
    """
    name: str
    use_re: bool = False
    bounds: np.ndarray = field(default_factory=utils.create_dummy_bounds)
    gprior: np.ndarray = field(default_factory=utils.create_dummy_gprior)
    re_var: float = np.inf

    def get_var_size(self, data: MRData) -> int:
        """Get optimization variable size.
        """
        return len(data.num_groups) if self.use_re else 1

    def get_cov_data(self, data: MRData) -> List[np.ndarray]:
        """Get the covariate.
        """
        assert self.name in data.covs
        cov = data.covs[self.name]
        if self.use_re:
            return utils.split_vec(cov, data.group_sizes)
        else:
            return [cov]

    def optvar2covmul(self,
                      x: np.ndarray,
                      group_sizes: np.ndarray) -> np.ndarray:
        """Converting optimization variable to covariate multiplier.
        """
        return np.repeat(x, group_sizes) if self.use_re else x

    def splitdata(self,
                  y: np.ndarray,
                  group_sizes: np.ndarray) -> List[np.ndarray]:
        """Split the data if use random effect.
        """
        return np.split(y, np.cumsum(group_sizes[:-1]))

    def objective_gprior(self,
                         x: np.ndarray,
                         group_sizes: np.ndarray) -> float:
        """Get the objective from the Gaussian prior for the total effects.
        """
        x = self.optvar2covmul(x, group_sizes)
        val = 0.0
        if np.isfinite(self.gprior[1]):
            val += 0.5*np.sum(((x - self.gprior[0])/self.gprior[1])**2)

        if self.use_re and np.isfinite(self.re_var):
            val += 0.5*((x - np.mean(x))**2)/self.re_var

        return val

    def predict(self,
                x: np.ndarray,
                covs: Dict[str, np.ndarray],
                group_sizes: np.ndarray) -> np.ndarray:
        """Predict with the optimization variable and provided covariates.
        """
        return covs[self.name]*self.optvar2covmul(x, group_sizes)

    def gradient(self,
                 x: np.ndarray,
                 covs: Dict[str, np.ndarray],
                 residual: np.ndarray,
                 group_sizes: np.ndarray) -> np.ndarray:
        """Compute the gradient of the optimization variable.
        """
        x = self.optvar2covmul(x, group_sizes)
        residual = self.splitdata(residual, group_sizes)
        cov = self.splitdata(covs[self.name], group_sizes)

        if self.use_re:
            grad = np.array(list(map(np.dot, cov, residual)))
        else:
            grad = np.array([np.dot(cov, residual)])

        if np.isfinite(self.gprior[1]):
            grad += (x - self.gprior[0])/self.gprior[1]**2

        if self.use_re and np.isfinite(self.re_var):
            grad += (x - np.mean(x))/self.re_var

        return grad


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

    def detach_data(self):
        """Detach the object from the data.
        """
        self.obs = None
        self.obs_se = None
        self.cov_data = None
        self.group_sizes = None
        self.var_sizes = None

    def objective(self, x: np.ndarray) -> float:
        """Optimization objective function.
        """
        # convert optimization variable to total effect for cov and group
        effects = utils.split_vec(x, self.var_sizes)
        # compute the prediction
        prediction = sum([
            utils.list_dot(effect, self.cov_data[i])
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
        effects = utils.split_vec(x, self.var_sizes)
        # compute the prediction
        prediction = sum([
            utils.list_dot(effect, self.cov_data[i])
            for i, effect in enumerate(effects)
        ])
        # compute scaled residual
        residual = (self.obs - prediction)/self.obs_se**2
        s_residual = utils.split_vec(residual, self.group_sizes)
        # gradient from the data likelihood
        grad = np.hstack([
            utils.list_dot(self.cov_data[i], s_residual) if cov.use_re else
            utils.list_dot(self.cov_data[i], [residual])
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
