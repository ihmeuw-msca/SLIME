# -*- coding: utf-8 -*-
"""
    test_optimizer
    ~~~~~~~~~~~~~~
"""
import numpy as np
import pytest
from slime.data import MRData
from slime.model import Covariate, MRModel
from slime.optimizer import SciOptLBFGSB


@pytest.fixture
def data():
    group = np.array(['A', 'A', 'B', 'B'])
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    covs = {'c1': np.arange(4), 'c2': np.random.randn(4)}
    obs_se = np.array([0.1, 0.2, 0.3, 0.4])

    return MRData(group=group,
                  obs=obs,
                  covs=covs,
                  obs_se=obs_se)


@pytest.fixture
def covariates():
    return [
        Covariate('intercept', use_re=True, re_var=1.0),
        Covariate('c1', gprior=np.array([0.0, 1.0])),
        Covariate('c2', bounds=np.array([1.0, 2.0]))
    ]


@pytest.fixture
def model(data, covariates):
    return MRModel(covariates, data=data)


def test_scioptlbfgs(model):
    optimizer = SciOptLBFGSB(model)
    optimizer.optimize()
