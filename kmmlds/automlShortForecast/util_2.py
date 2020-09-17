# -*- coding: utf-8 -*-

from datetime import timedelta

import numpy as np
import pandas as pd


def train_test_split(x_df, y_df, test_ratio=0.2):
    i_set = x_df['i_set']
    is_test = i_set > i_set.iloc[int(round((1 - test_ratio) * len(i_set)))]
    x_test = x_df.loc[is_test]
    y_test = y_df.loc[is_test]
    x_train = x_df.loc[~is_test]
    y_train = y_df.loc[~is_test]
    return x_train, y_train, x_test, y_test


def get_nwp_list(column_list):
    return tuple([col.split('_')[0] for col in column_list
                  if col.endswith('dist')])


def create_cv_index(i_set, k_folds, shuffle=False, seed=123):
    i_set_unique = np.unique(i_set)
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(i_set_unique)
    cv_index = np.ones_like(i_set, dtype=int) * -1
    for k in range(k_folds):
        for m in i_set_unique[k::k_folds]:
            cv_index[i_set == m] = k
    return cv_index


def create_cv_generator(i_set, k_folds, shuffle=False, seed=None):
    cv_index = create_cv_index(i_set, k_folds, shuffle=shuffle, seed=seed)
    for m in range(k_folds):
        is_cv = cv_index == m
        yield np.where(~is_cv)[0], np.where(is_cv)[0]


def add_feature_wdcut(x_df, n_sector, one_hot_encoding=True, concat=True):
    bins = np.arange(0, 360 + 1e-5, 360 / n_sector)
    bin_labels = ['s{}'.format(n) for n in range(n_sector)]
    nwp_list = get_nwp_list(x_df.columns)
    tmp_dict = {}
    for nwp in nwp_list:
        tmp_dict[nwp + '_wd_cut'] = pd.cut(x_df[nwp + '_wd'], bins, labels=bin_labels)
    df_wdcut = pd.DataFrame(tmp_dict)
    if one_hot_encoding:
        df_wdcut = pd.get_dummies(df_wdcut)
    if concat:
        return pd.concat([x_df, df_wdcut], axis=1)
    else:
        return df_wdcut


def add_feature_one_hot_horizon(x_df, concat=True):
    one_hot_df = pd.get_dummies(x_df['X_basic_horizon'], prefix='X_basic_horizon')
    if concat:
        return pd.concat([x_df, one_hot_df], axis=1)
    else:
        return one_hot_df


def add_feature_rho_crossed(x_df, col_name, concat=True):
    nwp_list = get_nwp_list(x_df.columns)
    tmp_dict = {}
    for nwp in nwp_list:
        if col_name == '_ws':
            tmp_dict[nwp + col_name + '_rho1'] = x_df[nwp + col_name] * x_df[nwp + '_rho']
            tmp_dict[nwp + col_name + '_rho2'] = x_df[nwp + col_name] ** 2 * x_df[nwp + '_rho']
            tmp_dict[nwp + col_name + '_angles'] = tmp_dict[nwp + col_name + '_rho1'] + tmp_dict[
                nwp + col_name + '_rho2']
        elif col_name == '_pre_ws':
            tmp_dict[nwp + col_name + '_rho1'] = x_df[nwp + col_name] * x_df[nwp + '_rho']
            tmp_dict[nwp + col_name + '_rho2'] = x_df[nwp + col_name] ** 2 * x_df[nwp + '_rho']
            tmp_dict[nwp + col_name + '_rho3'] = x_df[nwp + col_name] ** 3 * x_df[nwp + '_rho']
            tmp_dict[nwp + col_name + '_angles'] = tmp_dict[nwp + col_name + '_rho2'] + tmp_dict[
                nwp + col_name + '_rho3']
    tmp_df = pd.DataFrame(tmp_dict)
    if concat:
        return pd.concat([x_df, tmp_df], axis=1)
    else:
        return tmp_df


def add_feature_shift(df, param, shift, concat=True):
    if shift > 0:
        new_param = param + '_p{}'.format(shift)
    else:
        new_param = param + '_n{}'.format(abs(shift))
    new_df = df[['i_set', 'X_basic_time', param]].copy()
    new_df.columns = ['i_set', 'X_basic.time', new_param]
    new_df['X_basic_time'] = new_df['X_basic_time'] + timedelta(hours=shift)
    merged_df = pd.merge(df, new_df, how='left', on=['i_set', 'X_basic_time'])
    merged_df.index = df.index
    is_nan = merged_df[new_param].isnull()
    merged_df.loc[is_nan, new_param] = merged_df.loc[is_nan, param]
    if concat:
        return merged_df
    else:
        return merged_df[new_param]


def polar_to_cart(x_df, concat=True):
    nwp_list = get_nwp_list(x_df.columns)
    tmp_dict = {}
    for nwp in nwp_list:
        tmp_dict[nwp + '_u'] = x_df[nwp + '_ws'] * np.sin(x_df[nwp + '_wd'] / 180 * np.pi)
        tmp_dict[nwp + '_v'] = x_df[nwp + '_ws'] * np.cos(x_df[nwp + '_wd'] / 180 * np.pi)
    tmp_df = pd.DataFrame(tmp_dict)
    if concat:
        return pd.concat([x_df, tmp_df], axis=1)
    else:
        return tmp_df


def evaluation_rmse(df1, df2):
    return np.sqrt(np.nanmean((df1 - df2) ** 2))


def evaluation_mae(df1, df2):
    return np.abs(df1 - df2).mean().mean()
