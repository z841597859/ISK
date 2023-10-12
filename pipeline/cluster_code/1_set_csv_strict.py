import pandas as pd
import csv
from tqdm import tqdm

inputdir = '/media/c/D/zzw-c4/predict_raw_k'
outputdir = '/home/zzw/_test37_/Experiments/set_raw_k_with_i'

for fileindex in tqdm(range(256,512)):

    df = pd.read_csv(f'{inputdir}/emb_topredict_{fileindex}.csv')
    text = df['text'].to_list()
    id = df['id'].to_list()

    set_text=[]
    set_id=[]

    for  i,sent in enumerate(text):
        if sent not in set_text:
            set_text.append(sent)
            set_id.append(id[i])

    set_rows = [[set_text[i], set_id[i]] for i in range(len(set_text))]
    print(len(set_rows))

    with open(f'{outputdir}/set_raw_knowledge_{fileindex}.csv', 'w')as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'id'])
        writer.writerows(set_rows)

