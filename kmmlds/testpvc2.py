import argparse
import sys
from pathlib import Path


def main(args):
    parser = argparse.ArgumentParser(description='Returns sum of two arguments')
    parser.add_argument("--output", type=str, required=True)
    #parser.add_argument("--param2", type=str, required=True)

    args = parser.parse_args(args)
    

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    str1=''
    with open('/data/1/file1.txt') as f: 
       str1=f.read()        
    with open(args.output, 'w') as sum_path:
        sum_path.write(str1)
    


if __name__ == '__main__':
    main(sys.argv[1:])
