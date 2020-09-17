from datetime import datetime
import automlShortForecast.util as util
import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
#from automlShortForecast.automlShortForecast import automlShortForecast
from sklearn import linear_model
#from matplotlib import pyplot as plt
#import pickle
x_df=pd.read_csv('x.csv',index_col=0)
y_df=pd.read_csv('y.csv',index_col=0)

def feature_clean(x_df, y_df=None, cleaning_x=True):
    if cleaning_x:
        if isinstance(x_df, pd.DataFrame):
            is_valid = x_df.notnull().all(axis=1)
            #print(89, is_valid.shape)

        else:
            is_valid = x_df.notnull()
#             #print(92, is_valid.shape)
    else:
        is_valid = pd.Series(np.ones(len(x_df), dtype=bool), index=x_df.index)
    if y_df is not None:
        if isinstance(y_df, pd.DataFrame):
            is_valid = is_valid & y_df.notnull().all(axis=1)
        else:
            is_valid = is_valid & y_df.notnull()
    x_clean = x_df.loc[is_valid]
#     print(99, x_clean.shape)
#     print(100, is_valid)
#     print(101, )
    if y_df is not None:
        y_clean = y_df.loc[is_valid]
        return x_clean, y_clean, is_valid
    else:
        return x_clean, is_valid


def data_prepare(x_df):
    x_df = util.polar_to_cart(x_df)
    x_df = util.add_feature_wdcut(x_df, n_sector=8, one_hot_encoding=True)
    x_df = util.add_feature_rho_crossed(x_df, '.ws')
    return x_df


def _1st_stacking_feature(nwp):
    input_feature = [nwp + '.ws_rho1', nwp + '.ws_rho2', nwp + '.ws_angles', nwp + '.ws',
                     nwp + '.wd', nwp + '.rho', nwp + '.dist', nwp + '.u', nwp + '.v']
    input_feature.extend(['{}.wd_cut_s{}'.format(nwp, n) for n in range(8)])
    output_feature = nwp + '.pre_ws'
    return input_feature, output_feature

def LinearModel():
    return linear_model.LinearRegression()


def PreprocessingLinearModel(x_df, y_df,pickle_name='Linear_model.pkl'):
    x_df, y_df, _ = feature_clean(x_df, y_df, cleaning_x=True)
    #print(x_df)
    x_df = data_prepare(x_df)
    input_feature1, output_feature1 = _1st_stacking_feature('EC')
    input_feature2, output_feature2 = _1st_stacking_feature('GFS')
    input_feature=input_feature1+input_feature2
    model=LinearModel()
    fitted_model=model.fit(x_df[input_feature], y_df)
    y_pred=fitted_model.predict(x_df[input_feature])
    with mlflow.start_run():
        
#     with open(pickle_name, 'wb') as file:
#         pickle.dump(fitted_model, file)
#     plt.figure(figsize=(20,10))
#     plt.plot(y_pred)
#     plt.plot(y_df)
#     plt.show()

        # Log parameter, metrics, and model to MLflow
#         mlflow.log_param("alpha", alpha)
#         mlflow.log_param("l1_ratio", l1_ratio)
#         mlflow.log_metric("rmse", rmse)
#         mlflow.log_metric("r2", r2)
#         mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(model, "model")

    return fitted_model




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--masterid', type=str, default='', help='string')
    args = parser.parse_args()


#     print('masterid:', args.masterid, type(args.masterid))
#     get_token()
#     data={}
#     train(0.5,0.5,args.masterid)

    
    a=feature_clean(x_df,y_df)
    b=data_prepare(x_df)
    c=PreprocessingLinearModel(x_df, y_df,pickle_name='Linear_model.pkl')