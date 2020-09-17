import argparse
import sys
import json
import pandas as pd
from pathlib import Path

def transform():
    parser = argparse.ArgumentParser(description='Returns sum of two arguments')
    parser.add_argument("--list_data", type=str, required=True)
    #parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_list", type=str, required=True)
    args = parser.parse_args()
   
    with open(args.list_data,'r',encoding='utf8')as fp:
        json_data = json.load(fp)
        #print('这是文件中的json数据：',json_data)
        #print('这是读取到文件数据的数据类型：', type(json_data))

        
    
    #str0=json.loads(args.list_data)
   

    #str1=str0[0]

    df1=pd.DataFrame(json_data[0])
    df1.columns=["x_basic_time","ec_ws","gfs_wd","gfs_dist"]
    
    #df1=pd.DataFrame(str1)
    #df1.columns=["x_basic_time","ec_ws","gfs_wd","gfs_dist"]
    Path(args.output_list).parent.mkdir(parents=True, exist_ok=True)
    # with open(args.output_list, 'w') as f:
    #     f.write(df1)
    df1.to_csv(args.output_list,mode='a', index=False)    
if __name__ == "__main__":
    transform()
