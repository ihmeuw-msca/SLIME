# -*- coding: utf-8 -*-
"""
    test_mrdata
    ~~~~~~~~~~~
"""
import numpy as np
import pandas as pd
import pytest
from slime.data import MRData


def test_init():
    data = MRData()
    assert len(data.group) == 0
    assert len(data.obs) == 0
    assert len(data.obs_se) == 0
    assert len(data.covs) == 1
    assert 'intercept' in data.covs
    assert len(data.covs['intercept']) == 0

    assert data.num_obs == 0
    assert data.num_groups == 0
    assert len(data.groups) == 0
    assert len(data.group_sizes) == 0


@pytest.mark.parametrize('group', [np.array(['A', 'A', 'B', 'B'])])
@pytest.mark.parametrize('obs', [np.array([1.0, 2.0, 3.0, 4.0])])
@pytest.mark.parametrize('covs', [{'c1': np.arange(4),
                                   'c2': np.random.randn(4)}])
@pytest.mark.parametrize('obs_se', [None, np.array([0.1, 0.2, 0.3, 0.4])])
def test_init_wdata(group, obs, covs, obs_se):
    data = MRData(group=group,
                  obs=obs,
                  covs=covs,
                  obs_se=obs_se)

    assert data.num_obs == 4
    assert data.num_groups == 2
    assert (data.groups == np.array(['A', 'B'])).all()
    assert np.allclose(data.group_sizes, np.array([2, 2]))
    assert len(data.covs) == 3
    if obs_se is None:
        assert np.allclose(data.obs_se, 1.0)


@pytest.mark.parametrize('group', [np.array(['A', 'A', 'B', 'B'])])
@pytest.mark.parametrize('obs', [np.array([1.0, 2.0, 3.0, 4.0])])
def test_reset(group, obs):
    data = MRData(group=group,
                  obs=obs)
    data.reset()
    assert data.num_obs == 0
    assert data.num_groups == 0
    assert len(data.groups) == 0
    assert len(data.group_sizes) == 0
    assert len(data.covs) == 1


@pytest.fixture
def df():
    return pd.DataFrame({
        'group': np.array(['A', 'A', 'B', 'B']),
        'obs': np.array([1.0, 2.0, 3.0, 4.0]),
        'c1': np.arange(4),
        'c2': np.random.randn(4),
        'obs_se': np.array([0.1, 0.2, 0.3, 0.4])
    })


def test_load_df(df):
    data = MRData()
    data.load_df(df,
                 col_group='group',
                 col_obs='obs',
                 col_covs=['c1', 'c2'],
                 col_obs_se='obs_se')

    assert data.num_obs == 4
    assert data.num_groups == 2
    assert (data.groups == np.array(['A', 'B'])).all()
    assert np.allclose(data.group_sizes, np.array([2, 2]))
    assert len(data.covs) == 3
    assert np.allclose(data.obs_se, np.array([0.1, 0.2, 0.3, 0.4]))
