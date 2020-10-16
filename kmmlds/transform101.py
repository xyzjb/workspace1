import argparse
import sys
import json
import pandas as pd
from pyarrow import orc
from pathlib import Path

def transform():
    parser = argparse.ArgumentParser(description='Returns total lines of data records')
    parser.add_argument("--fileinput", type=str, required=True)
    #parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--totallines", type=str, required=True)
    args = parser.parse_args()
   



    filename=args.fileinput
    with open(filename,'rb') as file:
        data = orc.ORCFile(file)
        df = data.read().to_pandas()

    Path(args.totallines).parent.mkdir(parents=True, exist_ok=True)
    with open(args.totallines, 'w') as f:
        f.write(len(df))
        
if __name__ == "__main__":
    transform()
