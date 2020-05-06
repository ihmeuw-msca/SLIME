# -*- coding: utf-8 -*-
"""
    test_covariate
    ~~~~~~~~~~~~~~
"""
import numpy as np
import pytest
from slime.model import Covariate
from slime.data import MRData


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


@pytest.mark.parametrize('name', ['intercept'])
@pytest.mark.parametrize('use_re', [None, True, False])
@pytest.mark.parametrize('bounds', [None, np.array([0.0, 5.0])])
@pytest.mark.parametrize('gprior', [None, np.array([1.0, 1.0])])
@pytest.mark.parametrize('re_var', [None, 1.0])
def test_init(name, use_re, bounds, gprior, re_var):
    cov = Covariate(name,
                    use_re=use_re,
                    bounds=bounds,
                    gprior=gprior,
                    re_var=re_var)

    assert isinstance(cov.name, str)
    assert isinstance(cov.use_re, bool)
    assert isinstance(cov.bounds, np.ndarray)
    assert isinstance(cov.gprior, np.ndarray)
    assert isinstance(cov.re_var, float)


@pytest.mark.parametrize('name', ['intercept'])
@pytest.mark.parametrize('use_re', [True, False])
def test_get_var_size(data, name, use_re):
    cov = Covariate(name, use_re)
    if use_re:
        assert cov.get_var_size(data) == data.num_groups
    else:
        assert cov.get_var_size(data) == 1


@pytest.mark.parametrize('name', ['intercept', 'c1', 'c2'])
@pytest.mark.parametrize('use_re', [True, False])
def test_get_cov_data(data, name, use_re):
    cov = Covariate(name, use_re)
    if use_re:
        assert len(cov.get_cov_data(data)) == data.num_groups
    else:
        assert len(cov.get_cov_data(data)) == 1


@pytest.mark.parametrize('name', ['intercept'])
@pytest.mark.parametrize('use_re', [True, False])
def test_get_var_bounds(data, name, use_re):
    cov = Covariate(name, use_re)
    if use_re:
        assert len(cov.get_var_bounds(data)) == data.num_groups
    else:
        assert len(cov.get_var_bounds(data)) == 1
