from poseidon import poseidon
import time
import hashlib
import pandas as pd
import random
import argparse


    
    
# Model Tranining Sample
def train(in_alpha, in_l1_ratio, masterid):
    import os
    import warnings
    import sys

    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import ElasticNet

    import mlflow
    import mlflow.sklearn
    
    import logging
    ""
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)


    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2


    warnings.filterwarnings("ignore")
    np.random.seed(40)
    #print("xxxxxxxxxxxxxxxxxxxxx")
    # Read data from hive table prepared before
    data={}
    try:
        #print(masterid)
        #data=query_hive_data(masterid)[["sequence","x_basic_hour","x_basic_horizon","i_set","ec_ws","ec_wd","ec_tmp","ec_press","ec_rho","ec_dist","gfs_ws","gfs_wd","gfs_tmp","gfs_press","gfs_rho","gfs_dist","speed","power"]]
        data=df2.head(24962)
        data =data.fillna('1')
        #print(data)
    except Exception as e:
        logger.exception(
            "Get Hive Data Error: %s", e)

    print(data)    
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "power"
    train_x = train.drop(["power","speed"], axis=1)
    test_x = test.drop(["power","speed"], axis=1)
    train_y = train[["power"]]
    test_y = test[["power"]]
    
    
    # Set default values if no alpha is provided
    if float(in_alpha) is None:
        alpha = 0.5
    else:
        alpha = float(in_alpha)

    # Set default values if no l1_ratio is provided
    if float(in_l1_ratio) is None:
        l1_ratio = 0.5
    else:
        l1_ratio = float(in_l1_ratio)

    # Useful for multiple runs (only doing one run in this sample notebook)    
    with mlflow.start_run():
        # Execute ElasticNet
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        
        #print(test_x)
        # Evaluate Metrics
        predicted_powers = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_powers)

        # Print out metrics
#         print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
#         print("  RMSE: %s" % rmse)
#         print("  MAE: %s" % mae)
#         print("  R2: %s" % r2)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")

        #joblib.dump(lr, 'kongmingml.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--masterid', type=str, default='', help='string')
    parser.add_argument('--datasetname', type=str, default='', help='string')
    args = parser.parse_args()


    #print('masterid:', args.masterid, type(args.masterid))
    
#    x_df=pd.read_csv('x.csv',index_col=0)
#    y_df=pd.read_csv('y.csv',index_col=0) 
#
#
#    df1=pd.merge(x_df, y_df, how='inner', on=None, left_on=None, right_on=None,  
#          left_index=True, right_index=True)

#    df1['sequence']=df1.index
#    df1.head()
#    df1.rename(columns={'i.set':'i_set', 'X_basic.horizon':'x_basic_horizon', 'X_basic.hour':'x_basic_hour','EC.dist':'ec_dist', 'EC.ws':'ec_ws', 'EC.wd':'ec_wd',
#                   'EC.rho':'ec_rho', 'EC.pres':'ec_press', 'EC.tmp':'ec_tmp','GFS.dist':'gfs_dist', 'GFS.ws':'gfs_ws', 'GFS.wd':'gfs_wd',
#                   'GFS.rho':'gfs_rho', 'GFS.pres':'gfs_press', 'GFS.tmp':'gfs_tmp'}, inplace = True)
    dataset = Dataset.get_by_name(args.datasetname)
    df2=dataset.to_pandas_dataframe(use_cached_result=False)

   
    #get_token()
    data={}
    train(0.5,0.5,args.masterid)
