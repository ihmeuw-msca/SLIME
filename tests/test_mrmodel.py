# -*- coding: utf-8 -*-
"""
    test_mrmodel
    ~~~~~~~~~~~~
"""
import numpy as np
import pytest
from slime.data import MRData
from slime.model import Covariate, MRModel


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


def test_init(data, covariates):
    model = MRModel(covariates)
    model.attach_data(data)
    assert np.allclose(model.obs, data.obs)
    assert np.allclose(model.obs_se, data.obs_se)
    assert len(model.cov_data) == 3
    assert np.allclose(model.var_sizes, [2, 1, 1])
    assert np.allclose(model.group_sizes, data.group_sizes)
    assert len(model.var_bounds) == 4

    model.detach_data()
    assert model.obs is None
    assert model.obs_se is None
    assert model.cov_data is None
    assert model.group_sizes is None
    assert model.var_sizes is None
    assert model.var_bounds is None


@pytest.mark.parametrize('x', [np.ones(4), np.random.randn(4)])
def test_objective(data, covariates, x):
    model = MRModel(covariates, data=data)
    prediction = sum([
        data.covs['intercept']*np.repeat(x[:2], data.group_sizes),
        data.covs['c1']*x[2],
        data.covs['c2']*x[3]
    ])
    residual = data.obs - prediction
    tr_obj_val = 0.5*np.sum((residual/data.obs_se)**2)
    tr_obj_val += 0.5*np.sum((x[:2] - np.mean(x[:2]))**2)
    tr_obj_val += 0.5*x[2]**2
    assert np.isclose(model.objective(x), tr_obj_val)


@pytest.mark.parametrize('x', [np.random.randn(4)])
def test_gradient(data, covariates, x):
    model = MRModel(covariates, data=data)
    # complex step to compute the gradient
    finfo = np.finfo(float)
    step = finfo.tiny/finfo.eps
    x_c = x + 0j
    tr_grad = np.zeros(x.size)
    for i in range(x.size):
        x_c[i] += step*1j
        tr_grad[i] = model.objective(x_c).imag/step
        x_c[i] -= step*1j

    assert np.allclose(model.gradient(x), tr_grad)
