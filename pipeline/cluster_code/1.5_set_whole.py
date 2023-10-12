import pandas as pd
import csv
from tqdm import tqdm

def read_set_raw_knowledge(sentencesdir, file_num):
    df = pd.read_csv(f'{sentencesdir}/set_raw_knowledge_{file_num}.csv')
    text = df['text'].to_list()
    return text

i=0
sentencesdir=  '/home/zzw/_test37_/Experiments/set_raw_k_with_i'
sents = read_set_raw_knowledge(sentencesdir, i)

print('')

for sent in sents:
    print(sent)