import argparse
import sys
from pathlib import Path


def main(args):
    parser = argparse.ArgumentParser(description='Returns sum of two arguments')
    parser.add_argument("--file1", type=str, required=True)
    #parser.add_argument("--param2", type=str, required=True)

    args = parser.parse_args(args)

    Path('/tmp/data/1/file1.txt').parent.mkdir(parents=True, exist_ok=True)
    #path=Path('/data/1')
    #path.mkdir()
    with open('/tmp/data/1/file1.txt', 'w') as sum_path:
        sum_path.write(args.file1)
    


if __name__ == '__main__':
    main(sys.argv[1:])
