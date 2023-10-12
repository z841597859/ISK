import pickle

dir ='/home/zzw/_test37_/Experiments/result_and_sentences'

file_name='512-512-result-20-10-sents.pkl'

with open(f'{dir}/{file_name}', "rb") as fIn:
    stored_data = pickle.load(fIn)
    unique_communities = stored_data['unique_communities']

print(unique_communities[-1])