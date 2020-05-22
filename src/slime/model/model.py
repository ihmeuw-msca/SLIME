# -*- coding: utf-8 -*-
"""
    model
    ~~~~~
"""
import numpy as np
import scipy.optimize as sciopt
from slime.core import MRData
from .cov_model import CovModelSet
from typing import Dict, Any


class MRModel:
    """Linear MetaRegression Model.
    """

    def __init__(self, data: MRData, cov_models: CovModelSet):
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
        prediction = self.cov_models.predict(x)
        residual = self.obs - prediction
        return self.cov_models.gradient(x, residual, self.obs_se)

    def hessian(self, x: np.array) -> np.ndarray:
        """Hessian function for the optimtization.

        Args:
            x (np.array): optimization variable.

        Returns:
            np.ndarray: Hessian matrix.
        """
        return self.cov_models.hessian(x, self.obs_se)

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

    def sample_soln(self, num_draws: int = 1) -> Dict[Any, np.ndarray]:
        """Create draws for the solution.

        Args:
            num_draws (int, optional): Number of draws. Defaults to 1.

        Returns:
            Dict[Any, np.ndarray]:
                Dictionary with group_id as the key solution draws as the value.
        """
        if self.opt_result is None or self.result is None:
            RuntimeError('Fit the model first before sample the solution.')

        hessian = self.hessian(self.opt_result.x)
        info_mat = np.linalg.inv(hessian)

        samples = np.random.multivariate_normal(mean=self.opt_result.x,
                                                cov=info_mat,
                                                size=num_draws)
        _soln_samples = [
            self.cov_models.process_result(
                np.minimum(np.maximum(
                    samples[i], self.bounds[:, 0]), self.bounds[:, 1])
            )
            for i in range(num_draws)
        ]
        soln_samples = {
            g: np.vstack([
                _soln_samples[i][g]
                for i in range(num_draws)
            ])
            for g in self.cov_models.groups
        }
        return soln_samples
