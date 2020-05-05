# -*- coding: utf-8 -*-
"""
    data
    ~~~~
"""
from dataclasses import dataclass, field
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from .utils import empty_array


@dataclass
class MRData:
    """Data for simple linear mixed effects model.
    """
    group: np.ndarray = field(default_factory=empty_array)
    obs: np.ndarray = field(default_factory=empty_array)
    obs_se: np.ndarray = field(default_factory=empty_array)
    covs: Dict[str, np.ndarray] = field(default_factory=dict)

    num_obs: int = field(init=False, default=0)
    num_groups: int = field(init=False, default=0)
    groups: np.ndarray = field(init=False, default_factory=empty_array)
    group_sizes: np.ndarray = field(init=False, default_factory=empty_array)

    def __post_init__(self):
        self._get_num_obs()
        self._get_group_structure()
        self._add_intercept()
        self._add_obs_se()

        assert len(self.group) == self.num_obs
        assert len(self.obs) == self.num_obs
        assert len(self.obs_se) == self.num_obs
        assert all([len(self.covs[name]) == self.num_obs for name in self.covs])
        assert len(self.groups) == self.num_groups
        assert sum(self.group_sizes) == self.num_groups

    def _get_num_obs(self):
        """Get number of observation.
        """
        self.num_obs = len(self.obs)

    def _get_group_structure(self):
        """Get group structure.
        """
        self.groups, self.group_sizes = np.unique(self.group,
                                                  return_counts=True)
        self.num_groups = len(self.groups)

    def _add_intercept(self):
        """Add intercept.
        """
        self.covs['intercept'] = np.ones(self.num_obs)

    def _add_obs_se(self):
        """Add observation standard deviation.
        """
        if len(self.obs_se) == 0:
            self.obs_se = np.ones(self.num_obs)

    def reset_values(self):
        """Reset all the attributes to default values.
        """
        self.group = empty_array()
        self.obs = empty_array()
        self.obs_se = empty_array()
        self.covs = dict()
        self.__post_init__()

    def load_df(self, df: pd.DataFrame,
                col_group: str,
                col_obs: str,
                col_obs_se: Union[str, None] = None,
                col_covs: Union[List[str], None] = None):
        """Load data from data frame.
        """
        self.reset_values()
        self.group = df[col_group].to_numpy()
        self.obs = df[col_obs].to_numpy()
        if col_obs_se is not None:
            self.obs_se = df[col_obs_se].to_numpy()
        if col_covs is not None:
            self.covs = {
                col_cov: df[col_cov].to_numpy()
                for col_cov in col_covs
            }
        self.__post_init__()
