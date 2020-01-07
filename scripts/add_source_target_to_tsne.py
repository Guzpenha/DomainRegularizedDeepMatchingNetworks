import pandas as pd
from IPython import embed
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please input params: <path> <target>')
        exit(1)
    path = sys.argv[1]
    target = sys.argv[2]

    df = pd.read_csv(path)
    df['distribution'] = df.apply(lambda r: 'source' if r['domain'] != target else 'target', axis=1)
    df.to_csv(path)
    # embed()