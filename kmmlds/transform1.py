import argparse
import sys
import json
from pathlib import Path

def transform():
    parser = argparse.ArgumentParser(description='Returns sum of two arguments')
    parser.add_argument("--list_data", type=str, required=True)
    #parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_list", type=str, required=True)
    args = parser.parse_args()
   
    with open(args.list_data,'r',encoding='utf8')as fp:
        json_data = json.load(fp)
        str_data=json.dumps(json_data)

    str0=json.loads(str_data)
   
    abc=json.dumps(str0[1][0][0])
    bcd=json.dumps(str0[1][1][0])
   
    list2=json.loads("["+abc+","+bcd+"]")
    
    Path(args.output_list).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_list, 'w') as f:
        f.write(json.dumps(list2))
        
if __name__ == "__main__":
    transform()