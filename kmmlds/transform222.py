import argparse
import sys
import json
from pathlib import Path

def transform():
    parser = argparse.ArgumentParser(description='Returns sum of two arguments')
    parser.add_argument("--string_data", type=str, required=True)
    parser.add_argument("--datasetname", type=str, required=True)
    parser.add_argument("--output_list", type=str, required=True)
    args = parser.parse_args()
   
    
    str0=args.string_data
    str1=str0.upper()
    
    datasetname=args.datasetname
    
    #if str1=="WINDPOWERPREDICTION":
    #    str1='ABCDE0001'
   
    
    str2=f'["--masterid","\'{str1}\'","--datasetname","\'{datasetname}\'"]'

   
    #list2=json.loads("["+abc+","+bcd+"]")
    Path(args.output_list).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_list, 'w') as f:
        f.write(str2)
        
if __name__ == "__main__":
    transform()
