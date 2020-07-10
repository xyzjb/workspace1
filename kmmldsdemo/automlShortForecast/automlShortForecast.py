# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

from . import util
from azureml.core import Workspace, get_run
from .xgboost_utils import cross_prediction
from .xgboost_utils import feature_clean
from .xgboost_utils import hyperparameter_search
from azureml.core import Run
from azureml.train.automl import AutoMLConfig
from azureml.core.experiment import Experiment
from azureml.train.automl.run import AutoMLRun
from azureml.core import run
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from azureml.core.authentication import InteractiveLoginAuthentication
import azureml.train.automl.runtime

import os
from azureml.core.model import Model

class automlShortForecast(object):

    def __init__(self, pc, nwp_list, n_sector=8, k_fold=5, n_iter=30, max_trees=300, ):
        self.pc = pc
        self.nwp_list = nwp_list
        self.n_sector = n_sector
        self.k_fold = k_fold
        self.n_iter = n_iter
        self.max_trees = max_trees
        self._estimator_ = {}
        self.grid_params = {
            'silent': [1],
            'eta': [0.05],
            'max_depth': range(3, 6),
            'min_child_weight': [1, 3, 10],
            'subsample': [0.5, 0.6, 0.7, 0.8],
            'lambda': [0.1, 0.3, 1, 3, 10]
        }

    @classmethod
    def create_basic_feature(cls, obs_data, nwp_data, starts, horizon):
        columns = 'i.set X_basic.forecast_time X_basic.horizon X_basic.time X_basic.hour'.split()
        for nwp in nwp_data.keys():
            columns.extend([f'{nwp}.{attr}' for attr in 'nwp_time dist ws wd rho pres tmp'.split()])
        repeats = horizon + 1
        days = len(starts)
        feat = pd.DataFrame(data=np.nan, index=range(repeats * days), columns=columns)

        feat['i.set'] = np.repeat(range(days), repeats)
        feat['X_basic.forecast_time'] = np.repeat(starts, repeats)
        feat['X_basic.horizon'] = np.tile(range(repeats), days)
        feat['X_basic.time'] = feat['X_basic.forecast_time'] + pd.to_timedelta(feat['X_basic.horizon'], unit='h')
        feat['X_basic.hour'] = feat['X_basic.time'].dt.hour

        for nwp in nwp_data.keys():
            feat[f'{nwp}.nwp_time'] = np.repeat(starts - pd.Timedelta(hours=12), repeats)
            feat[f'{nwp}.dist'] = (feat['X_basic.time'] - feat[f'{nwp}.nwp_time']).dt.total_seconds() / 3600
            for attr in ('WS', 'WD', 'RHO', 'PRES', 'TMP'):
                feat[f'{nwp}.{attr.lower()}'] = nwp_data[nwp][attr].values

        y = pd.DataFrame()
        # feat['X_basic.time'] = feat['X_basic.time'].astype(str)
        y['speed'] = obs_data['speed'][feat['X_basic.time']].values
        y['power'] = obs_data['power'][feat['X_basic.time']].values

        return feat, y

    def autoMLRegression(self, x_df, y_df):
        run = Run.get_context()
        experiment = run.experiment
        train_data = pd.concat([x_df, y_df], axis=1)
        column_name = list(y_df)[0]
        automl_classifier = AutoMLConfig(
            task='regression',
            primary_metric='normalized_root_mean_squared_error',
            experiment_timeout_minutes=15,
            training_data=train_data,
            label_column_name=column_name,
            n_cross_validations=self.k_fold,
            enable_onnx_compatible_models=True,
            model_explainability=True)
        run = experiment.submit(automl_classifier, show_output=True)
        best_run, fitted_model = run.get_output()
        return best_run, fitted_model

    def existingModel(self, exp_name, run_id):
        SUBSCRIPTION_ID = '1f6fddae-bfa7-4f33-b9a5-ad3d4f29b8a9'
        RESOURCE_GROUP = 'DECADAAPPS'
        WORKSPACE_NAME = 'kongming-aml'
        TENANT_ID = 'd7802200-0ab3-48a9-a946-c4e20d15c1ca'

        auth = InteractiveLoginAuthentication(tenant_id=TENANT_ID)
        ws = Workspace(subscription_id=SUBSCRIPTION_ID,
               resource_group=RESOURCE_GROUP,
               workspace_name=WORKSPACE_NAME,
               auth=auth)
        exp = Experiment(ws, exp_name)
        run = AutoMLRun(experiment=exp, run_id=run_id)
        _, model = run.get_output()
        return run, model

    def fit(self, x_df, y_df):
        x_df, y_df, _ = feature_clean(x_df, y_df, cleaning_x=True)
        x_df = self.data_prepare(x_df)
        # stacking: 1st layer
        new_feature = []

        for nwp in self.nwp_list:
            input_feature, output_feature = self._1st_stacking_feature(nwp)
            # if nwp == 'EC':
            #     best_run, fitted_model = self.existingModel('EXPERIMENT_NAME_0', 'AutoML_91a6182d-379c-4c96-8dd4-b01a775304d4')
            # else:
            #     best_run, fitted_model = self.existingModel('EXPERIMENT_NAME_0','AutoML_d6f3fe15-fcde-4ed9-a1bb-68a28415c8a3')
            best_run, fitted_model = self.autoMLRegression(x_df[input_feature], y_df[['speed']])
            best_run.register_model(model_name=f'first_layer_{nwp}', model_path=f'outputs/model.onnx',
                                    model_framework='Onnx')
            result = fitted_model.predict(x_df[input_feature])

            pre_ws = pd.Series(result, index=x_df.index, name=output_feature)
            pre_pw = np.interp(pre_ws, self.pc['speed'], self.pc['power'])
            new_feature.append(pd.Series(pre_pw, index=x_df.index, name=nwp + '.WSPV'))
            new_feature.append(pre_ws)
        x_df = pd.concat([x_df, *new_feature], axis=1)
        # shift meta result
        x_df = util.add_feature_rho_crossed(x_df, '.pre_ws')
        x_df = self._shift_metaresult(x_df, '.pre_ws_rho3')

        # stacking: 2nd layer
        new_feature = []
        for nwp in self.nwp_list:
            input_feature, output_feature = self._2nd_stacking_feature(nwp)
            best_run, fitted_model = self.autoMLRegression(x_df[input_feature], y_df[['power']])
            best_run.register_model(model_name = f'second_layer_{nwp}', model_path=f'outputs/model.onnx', model_framework='Onnx')
            # if nwp == 'EC':
            #     best_run, fitted_model = self.existingModel('EXPERIMENT_NAME_0', 'AutoML_9b1b418f-0f43-40c9-93f0-a2f69e506ed5')
            # else:
            #     best_run, fitted_model = self.existingModel('EXPERIMENT_NAME_0','AutoML_1dd44f64-833c-4019-b616-026930fa5ab1')
            result = fitted_model.predict(x_df[input_feature])
            new_feature.append(pd.Series(result, index=x_df.index, name=output_feature))
        x_df = pd.concat([x_df, *new_feature], axis=1)
        # linear regression
        input_feature, output_feature = self._final_fusion_feature()
        result = self._build_linear_regression(x_df[input_feature], y_df['power'], x_df['X_basic.horizon'])
        return pd.Series(result, index=x_df.index, name=output_feature)

    def data_prepare(self, x_df):
        x_df = util.polar_to_cart(x_df)
        x_df = util.add_feature_wdcut(x_df, n_sector=self.n_sector, one_hot_encoding=True)
        x_df = util.add_feature_rho_crossed(x_df, '.ws')
        return x_df

    def _1st_stacking_feature(self, nwp):
        input_feature = [nwp + '.ws_rho1', nwp + '.ws_rho2', nwp + '.ws_angles', nwp + '.ws',
                         nwp + '.wd', nwp + '.rho', nwp + '.dist', nwp + '.u', nwp + '.v']
        input_feature.extend(['{}.wd_cut_s{}'.format(nwp, n) for n in range(self.n_sector)])
        output_feature = nwp + '.pre_ws'
        return input_feature, output_feature

    def _build_xgb(self, x_df, y_df, i_fold, use_column_name=False):
        x_clean, y_clean, is_valid = feature_clean(x_df, y_df)
        assert y_clean.empty is False
        i_fold = i_fold[is_valid]
        dtrain = xgb.DMatrix(x_clean.loc[i_fold != 0], y_clean.loc[i_fold != 0], missing=0)
        ddev = xgb.DMatrix(x_clean.loc[i_fold == 0], y_clean.loc[i_fold == 0], missing=0)
        lscore, lntree, lparam, _ = hyperparameter_search(dtrain, ddev, self.grid_params, n_iter=self.n_iter,
                                                          verbose_eval=False, early_stopping_rounds=8,
                                                          num_boost_round=self.max_trees)
        idx_min = np.argmin(lscore)
        tmp_result, bst = cross_prediction(lparam[idx_min], x_clean, y_clean, i_fold, lntree[idx_min], return_bst=True,
                                           use_column_name=use_column_name)
        result = np.nan * np.zeros(len(y_df))
        result[is_valid] = tmp_result
        return result, bst

    def _shift_metaresult(self, df, column_name):
        shift_feature = []
        for nwp in self.nwp_list:
            for shift_index in range(1, 4):
                shift_feature.append(util.add_feature_shift(df, nwp + column_name, shift_index, concat=False))
                shift_feature.append(util.add_feature_shift(df, nwp + column_name, -shift_index, concat=False))
        return pd.concat([df, *shift_feature], axis=1)

    @staticmethod
    def _2nd_stacking_feature(nwp):
        input_feature = [nwp + '.WSPV', nwp + '.pre_ws', nwp + '.pre_ws_rho1', nwp + '.pre_ws_rho2',
                         nwp + '.pre_ws_rho3', nwp + '.pre_ws_angles',
                         nwp + '.pre_ws_rho3_p1', nwp + '.pre_ws_rho3_p2', nwp + '.pre_ws_rho3_p3',
                         nwp + '.pre_ws_rho3_n1', nwp + '.pre_ws_rho3_n2', nwp + '.pre_ws_rho3_n3',
                         ]
        output_feature = nwp + '.wp_fcst'
        return input_feature, output_feature

    def _final_fusion_feature(self, nwp_list=()):
        if len(nwp_list) == 0:
            input_feature = [nwp + '.wp_fcst' for nwp in self.nwp_list]
        else:
            input_feature = [nwp + '.wp_fcst' for nwp in nwp_list]
        output_feature = 'final_fcst'
        return input_feature, output_feature

    def _build_linear_regression(self, x_df, y_df, horizon_arr, key_suffix=''):
        run = Run.get_context()
        experiment = run.experiment
        ws = experiment.workspace
        if x_df.shape[1] == 1:
            return x_df.values.ravel()
        _, _, is_valid = feature_clean(x_df, y_df)
        horizon_list = horizon_arr.unique()
        model_dict = {}
        result = np.zeros(len(y_df)) * np.nan
        initial_type = [('float_input', FloatTensorType([None, 2]))]
        for horizon in horizon_list:
            linear_model = Ridge(fit_intercept=False)
            linear_model.fit(x_df.loc[is_valid & (horizon_arr == horizon)], y_df[is_valid & (horizon_arr == horizon)])
            onx = convert_sklearn(linear_model, initial_types=initial_type)
            os.makedirs('outputs', exist_ok=True)
            with open(f'outputs/model_{horizon}.onnx', "wb+") as f:
                f.write(onx.SerializeToString())
            model = Model.register(workspace=ws,
                                   model_path=f'outputs/model_{horizon}.onnx',
                                   model_name=f"linear_{horizon}")
            model_dict[horizon] = linear_model
            result[is_valid & (horizon_arr == horizon)] = linear_model.predict(
                x_df.loc[is_valid & (horizon_arr == horizon)])
        if len(key_suffix) == 0:
            self._estimator_['fusion'] = model_dict
        else:
            self._estimator_['fusion' + '_' + key_suffix] = model_dict
        return result
