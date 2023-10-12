import pandas as pd
import csv
from tqdm import tqdm

dir = '/home/zzw/_test37_/Experiments/set_raw_k_with_i'
outputdir = '/home/zzw/_test37_/EXEVAL'

rows=[]
count=0
start = 0
end = 64

for fileindex in tqdm(range(start,end)):

    df = pd.read_csv(f'{dir}/set_raw_knowledge_{fileindex}.csv')
    text = df['text'].to_list()
    # id = df['id'].to_list()

    rows+=text
    count += len(text)

print(len(rows))
print(count)

with open(f'{outputdir}/set_k_collect_64.txt','w') as f:

    for row in rows:
        f.write(row)
        f.write('\n')
