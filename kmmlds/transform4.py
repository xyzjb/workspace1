import argparse
import sys
import json
import datetime

def transform():
    parser = argparse.ArgumentParser(description='Returns sum of two arguments')
    parser.add_argument("--list_data", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--number_data", type=str, required=True)
    args = parser.parse_args()
   
    
    str0=json.loads(args.list_data)
    current_date=datetime.datetime.now().strftime('%Y-%m-%d')
    str1=""
    for item in str0[1]:
        masterid=item[0]
        str1=str1+json.dumps(masterid).replace('\"','')+current_date+"1,"

    
    with open(args.output_path, 'w') as f:
        f.write("["+str1[:-1]+"]")
        
if __name__ == "__main__":
    transform()
    
