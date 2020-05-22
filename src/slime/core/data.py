# -*- coding: utf-8 -*-
"""
    data
    ~~~~
"""
import numpy as np
from . import utils


class MRData:
    """Data used for fitting the simple linear mixed effects model.
    """

    def __init__(self, df, col_group, col_obs, col_obs_se=None, col_covs=None):
        """Constructor of the ODEData.
        Args:
            df (pd.DataFrame): Dataframe contains data.
            col_group (str): Name of the group column.
            col_obs (str): Name of the observation column.
            col_obs_se (str | None, optional):
                Name of the observation standard error.
            col_covs (list{str} | None, optional): Names of the covariates.
        """
        self.df_original = df.copy()
        self.col_group = col_group
        self.col_obs = col_obs
        self.col_obs_se = col_obs_se
        self.col_covs = [] if col_covs is None else col_covs

        # add intercept as default covariates
        df['intercept'] = 1.0
        if 'intercept' not in self.col_covs:
            self.col_covs.append('intercept')

        # add observation standard error
        if self.col_obs_se is None:
            self.col_obs_se = 'obs_se'
            df[self.col_obs_se] = 1.0

        assert self.col_group in df
        assert self.col_obs in df
        assert self.col_obs_se in df
        assert all([name in df for name in self.col_covs])
        self.df = df[[self.col_group, self.col_obs, self.col_obs_se] +
                     self.col_covs].copy()
        self.df.sort_values(col_group, inplace=True)
        self.groups, self.group_sizes = np.unique(self.df[self.col_group],
                                                  return_counts=True)

        self.group_idx = utils.sizes_to_indices(self.group_sizes)
        self.num_groups = len(self.groups)
        self.num_obs = self.df.shape[0]

    def df_by_group(self, group):
        """Divide data by group.
        Args:
            group (any): Group id in the data frame.
        Returns:
            pd.DataFrame: The corresponding data frame.
        """
        assert group in self.groups
        return self.df[self.df[self.col_group] == group]
