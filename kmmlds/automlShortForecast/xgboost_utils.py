# -*- coding: utf-8 -*-

import operator
from functools import reduce

import numpy as np
import pandas as pd
import xgboost as xgb


def hyperparameter_search(dtrain, dtest, grid_params, n_iter=32, seed=None, num_boost_round=300,
                          early_stopping_rounds=8, verbose_eval=False, **kwargs):
    if seed is not None:
        np.random.seed(seed)
    param_list, score_list, best_ntree_list, importance_by_gain = [], [], [], []
    for param in _ParameterSampler(grid_params, n_iter):
        bst = xgb.train(param, dtrain, num_boost_round, evals=[(dtrain, 'train'), (dtest, 'eval')],
                        verbose_eval=verbose_eval, early_stopping_rounds=early_stopping_rounds, **kwargs)
        param_list.append(param)
        score_list.append(bst.best_score)
        best_ntree_list.append(bst.best_ntree_limit)
        importance_by_gain.append(
            sorted(bst.get_score(importance_type='gain').items(), key=lambda k: k[1], reverse=True))
    return score_list, best_ntree_list, param_list, importance_by_gain


def score_stats_by_params(score_list, params_list):
    arr_score = pd.Series(score_list)
    df_params = pd.DataFrame(params_list)
    result = []
    for param in df_params.columns:
        result.append(arr_score.groupby(df_params[param]).agg(['mean', 'std']))
    return pd.concat(result, keys=df_params.columns)


def get_self_or_values(df, use_column_name):
    return df if use_column_name else df.values


def cross_prediction(params, x_df, y_df, idx_fold, num_boost_round, missing=None, return_bst=False,
                     use_column_name=False):
    prediction = np.zeros_like(idx_fold, dtype=float) * np.nan
    for idx in np.unique(idx_fold):
        dtrain = xgb.DMatrix(get_self_or_values(x_df.loc[idx_fold != idx], use_column_name),
                             label=get_self_or_values(y_df.loc[idx_fold != idx], use_column_name), missing=missing)
        dtest = xgb.DMatrix(get_self_or_values(x_df.loc[idx_fold == idx], use_column_name), missing=missing)
        bst = xgb.train(params, dtrain, num_boost_round)
        prediction[idx_fold == idx] = bst.predict(dtest)
    if return_bst:
        dtrain = xgb.DMatrix(get_self_or_values(x_df, use_column_name), label=get_self_or_values(y_df, use_column_name),
                             missing=missing)
        bst = xgb.train(params, dtrain, num_boost_round)
        return prediction, bst
    else:
        return prediction


class _ParameterSampler(object):
    def __init__(self, param_distribution, n_iter):
        self.param_distr = param_distribution
        self.n_iter = n_iter
        self.p = None
        if isinstance(self.param_distr, list):
            n_sets = []
            for param_dict in self.param_distr:
                n_sets.append(reduce(operator.mul, [len(x) for x in param_dict.values()]))
            self.p = np.array(n_sets) / sum(n_sets)

    def __iter__(self):
        for kk in range(self.n_iter):
            if isinstance(self.param_distr, list):
                i_grid = np.random.choice(range(len(self.param_distr)), p=self.p)
                yield self.random_sample(self.param_distr[i_grid])
            else:
                yield self.random_sample(self.param_distr)

    @staticmethod
    def random_sample(params_grid):
        result = {}
        for k, v in params_grid.items():
            result[k] = np.random.choice(v)
        return result


def feature_clean(x_df, y_df=None, cleaning_x=True):
    if cleaning_x:
        if isinstance(x_df, pd.DataFrame):
            is_valid = x_df.notnull().all(axis=1)
            print(89, is_valid.shape)

        else:
            is_valid = x_df.notnull()
            print(92, is_valid.shape)
    else:
        is_valid = pd.Series(np.ones(len(x_df), dtype=bool), index=x_df.index)
    if y_df is not None:
        if isinstance(y_df, pd.DataFrame):
            is_valid = is_valid & y_df.notnull().all(axis=1)
        else:
            is_valid = is_valid & y_df.notnull()
    x_clean = x_df.loc[is_valid]
    print(99, x_clean.shape)
    print(100, is_valid)
    print(101, )
    if y_df is not None:
        y_clean = y_df.loc[is_valid]
        return x_clean, y_clean, is_valid
    else:
        return x_clean, is_valid
