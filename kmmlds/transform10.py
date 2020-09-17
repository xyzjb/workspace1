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
   
    
    str0=json.loads(args.list_data)
   
#    abc=json.dumps(str0[0][0][0])
#    bcd=json.dumps(str0[0][1][0])
    str1=str0[0]
    #list2=json.loads(str0)
    
    df1=pd.DataFrame(str1)
    df1.columns=["x_basic_time","ec_ws","gfs_wd","gfs_dist"]
    Path(args.output_list).parent.mkdir(parents=True, exist_ok=True)
    # with open(args.output_list, 'w') as f:
    #     f.write(df1)
    df1.to_csv(args.output_list,mode='a', index=False)    
if __name__ == "__main__":
    transform()

