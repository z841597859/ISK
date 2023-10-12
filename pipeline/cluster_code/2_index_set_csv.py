import pandas as pd
import csv
from tqdm import tqdm

dir = '/home/zzw/_test37_/Experiments/set_raw_k_with_i'

start = 0
end = 512

rows=[]
count=0
for fileindex in tqdm(range(start,end)):

    df = pd.read_csv(f'{dir}/set_raw_knowledge_{fileindex}.csv')
    text = df['text'].to_list()
    id = df['id'].to_list()

    rows.append([fileindex,count])

    count += len(text)

with open(f'{dir}/index_for_set_raw_knowledge_{end}.csv', 'w')as f:
    writer = csv.writer(f)
    writer.writerow(['file_num','start_index'])
    writer.writerows(rows)

df = pd.read_csv(f'{dir}/index_for_set_raw_knowledge_{end}.csv')
file_num = df['file_num'].to_list()
start_index = df['start_index'].to_list()

print(file_num)
print(start_index)