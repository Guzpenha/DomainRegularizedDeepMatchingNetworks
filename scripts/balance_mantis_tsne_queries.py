import pandas as pd
from IPython import embed
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please input params: <tnse.csv> file')
        exit(1)
    path = sys.argv[1]

    df = pd.read_csv(path)
    filtered_df = []
    i=0
    for idx, r in df.iterrows():
        if(r['domain']=='askubuntu' and i < 700):
            i+=1
            filtered_df.append(r)
        elif(r['domain']!='askubuntu'):
            filtered_df.append(r)
    filtered_df = pd.DataFrame(filtered_df)
    filtered_df.to_csv(path+"_filtered")
    # embed()