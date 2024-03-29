import argparse
import sys
import json
from pathlib import Path

def transform():
    parser = argparse.ArgumentParser(description='Returns sum of two arguments')
    parser.add_argument("--list_data", type=str, required=True)
    #parser.add_argument("--output_path", type=str, required=True)
    #parser.add_argument("--number_data", type=str, required=True)
    parser.add_argument("--list_output1", type=str, required=True)
    args = parser.parse_args()
   
    
    with open(args.list_data,'r',encoding='utf8')as fp:
        json_data = json.load(fp)
        str_data=json.dumps(json_data)

    
    
    
    
    #if args.list_data != '[[], []]':
    if str_data != '[[], []]':
        str0=json.loads(str_data)
   
        masterid=json.dumps(str0[1][0][0]).replace('\"','')
        str_date=json.dumps(str0[1][0][1]).replace('\"','')
        str_flag=json.dumps(str0[1][0][2])
        str_output=masterid+str_date+str_flag
        #str_output=masterid.lower()
    else:
        str_output="NNNNN"
        
    Path(args.list_output1).parent.mkdir(parents=True, exist_ok=True)
    with open(args.list_output1, 'w') as f:
        f.write(str_output)
        
if __name__ == "__main__":
    transform()