import pickle
import pandas as pd
from tqdm import tqdm

sentencesdir = '/home/zzw/_test37_/Experiments/set_raw_k_with_i'
dir = '/home/zzw/_test37_/Experiments/result_and_sentences'
whole_file_num=512
result_name='512-512-result-20-10'

def read_set_raw_knowledge(sentencesdir, file_num):
    df = pd.read_csv(f'{sentencesdir}/set_raw_knowledge_{file_num}.csv')
    text = df['text'].to_list()
    return text


with open(f'{dir}/{result_name}.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    unique_communities = stored_data['unique_communities']

sents=[]
for i in tqdm(range(0,whole_file_num)):
    sents_ = read_set_raw_knowledge(sentencesdir,i)
    sents += sents_

for id,cluster in tqdm(enumerate(unique_communities)):
    unique_communities[id] = [ sents[j] for j in cluster ]

with open(f'{dir}/{result_name}-sents.pkl',"wb") as fOut:
    pickle.dump({'unique_communities': unique_communities},fOut, protocol=pickle.HIGHEST_PROTOCOL)

