# -*- coding: utf-8 -*-
"""
    cov_model
    ~~~~~~~~~
"""
import numpy as np
from scipy.linalg import block_diag
from slime.core import MRData
import slime.core.utils as utils


class CovModel:
    """Single covariate model.
    """

    def __init__(self, col_cov,
                 use_re=False,
                 bounds=(-np.inf, np.inf),
                 gprior=(0.0, np.inf),
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
        self.bounds = np.array(bounds)
        self.gprior = np.array(gprior)
        self.re_var = re_var

        self.name = self.col_cov
        self.var_size = None

        self.cov = None
        self.cov_mat = None
        self.cov_scale = None

        self.group_idx = None
        self.group_sizes = None

    def attach_data(self, data: MRData):
        """Attach the data.

        Args:
            data (MRData): MRData object.
        """
        self.group_idx = data.group_idx
        self.group_sizes = data.group_sizes
        assert self.col_cov in data.df
        if self.use_re:
            self.var_size = data.num_groups
        else:
            self.var_size = 1

        cov = data.df[self.col_cov].values
        cov_scale = np.linalg.norm(cov)
        assert cov_scale > 0.0
        self.cov = cov/cov_scale
        self.cov_scale = cov_scale
        if self.use_re:
            self.cov_mat = block_diag(*[
                self.cov[self.group_idx[i]][:, None]
                for i in range(data.num_groups)
            ])
        else:
            self.cov_mat = self.cov[:, None]

    def detach_data(self):
        """Detach the object from the data.
        """
        self.var_size = None
        self.cov = None
        self.cov_mat = None
        self.cov_scale = None
        self.group_sizes = None
        self.group_idx = None

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
        return self.cov*self.get_cov_multiplier(x)

    def prior_objective(self, x):
        """Objective related to prior.

        Args:
            x (np.ndarray): optimization variable.
        """
        val = 0.0

        x = x/self.cov_scale
        # random effects priors
        if self.use_re and np.isfinite(self.re_var):
            val += 0.5*np.sum((x - np.mean(x))**2)/self.re_var

        # Gaussian prior for the effects
        if np.isfinite(self.gprior[1]):
            val += 0.5*np.sum(((x - self.gprior[0])/self.gprior[1])**2)

        return val

    def gradient(self, x, residual, obs_se):
        """Compute the gradient for the covariate multiplier.

        Args:
            x (np.ndarray): optimization variable.
            residual (np.ndarray): residual array, observation minus prediction.
            obs_se (np.ndarray): observation standard deviation.

        Return:
            np.ndarray: gradient
        """
        grad = -(self.cov_mat.T/obs_se).dot(residual/obs_se)
        x = x/self.cov_scale
        if self.use_re and np.isfinite(self.re_var):
            grad += ((x - np.mean(x))/self.re_var)/self.cov_scale

        if np.isfinite(self.gprior[1]):
            grad += ((x - self.gprior[0])/self.gprior[1]**2)/self.cov_scale

        return grad

    def extract_bounds(self):
        """Extract the bounds for the optimization problem.
        """
        bounds = self.bounds*self.cov_scale
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
        self.num_covs = len(self.cov_models)

        self.var_size = None
        self.var_sizes = None
        self.var_idx = None
        self.groups = None
        self.num_groups = None

        if data is not None:
            self.attach_data(data)

    def attach_data(self, data):
        """Attach the data.

        Args:
            data (MRData): MRData object.
        """
        for cov_model in self.cov_models:
            cov_model.attach_data(data)

        self.var_sizes = np.array([
            cov_model.var_size for cov_model in self.cov_models
        ])
        self.var_size = np.sum(self.var_sizes)
        self.var_idx = utils.sizes_to_indices(self.var_sizes)
        self.groups = data.groups
        self.num_groups = data.num_groups

    def detach_data(self):
        """Detach the object from the data.
        """
        for cov_model in self.cov_models:
            cov_model.detach_data()

        self.var_size = None
        self.var_sizes = None
        self.var_idx = None

        self.groups = None
        self.num_groups = None

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

    def gradient(self, x, residual, obs_se):
        """Gradient function.

        Args:
            x (np.ndarray): optimization variable.
            residual (np.ndarray): residual array, observation minus prediction.
            obs_se (np.ndarray): observation standard deviation.

        Return:
            np.ndarray: gradient
        """
        return np.hstack([
            cov_model.gradient(x[self.var_idx[i]], residual, obs_se)
            for i, cov_model in enumerate(self.cov_models)
        ])

    def hessian(self, x: np.array, obs_se: np.array) -> np.ndarray:
        """Hessian function.

        Args:
            x (np.array): optimization variable.
            obs_se (np.array): observation standard error.

        Returns:
            np.ndarray: Hessian matrix.
        """
        cov_mat = np.hstack([
            cov_model.cov_mat
            for cov_model in self.cov_models
        ])
        prior_diag = np.hstack([
            np.repeat(1.0/cov_model.gprior[1]**2, cov_model.var_size)
            for cov_model in self.cov_models
        ])
        return (cov_mat.T/obs_se**2).dot(cov_mat) + np.diag(prior_diag)

    def extract_bounds(self):
        """Extract the bounds for the optimization problem.
        """
        return np.vstack([
            cov_model.extract_bounds()
            for cov_model in self.cov_models
        ])

    def process_result(self, x):
        """Process the result, organize it by group and scale by the
        cov_scale.

        Args:
            x (np.ndarray): optimization variable.
        """
        coefs = np.vstack([
            x[self.var_idx[i]]/cov_model.cov_scale if cov_model.use_re else
            np.repeat(x[self.var_idx[i]], self.num_groups)/cov_model.cov_scale
            for i, cov_model in enumerate(self.cov_models)
        ])
        return {
            g: coefs[:, i]
            for i, g in enumerate(self.groups)
        }
