import torch
import pickle
import itertools
from tqdm import tqdm

dir = '/media/c/D/zzw-c4/cluster_data_with_i'
outputdir = '/media/c/D/zzw-c4/cluster_ex_with_r_s'
device = 'cuda:1'
# stored_topk = 100

start=0
end=256

def read_one_cluster_data(dir,file_index,device):
    with open(f'{dir}/4_cluster_{file_index}.pkl',"rb") as fIn:
        stored_data = pickle.load(fIn) #device while writepkl
        k_values = stored_data['k_values'].to(device)
        k_ids = stored_data['k_ids'].to(device)
    return k_values, k_ids

def extract_one_cluster_data(topk_v, topk_i, device, threshold=0.75, min_community_size=20):
    threshold = torch.tensor(threshold, device=device)
    extracted_communities = []
    for i in range(len(topk_v)):
        new_cluster = []
        topk_i_list = topk_i[i].tolist()
        for j in range(topk_v.shape[1]):
            if topk_v[i][j] >= threshold:
                new_cluster.append(topk_i_list[j])
            else:
                break
        if len(new_cluster) >= min_community_size:
            extracted_communities.append(new_cluster)

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # for i in range(len(topk_v)):
    #     scores = 0
    #     new_cluster = []
    #     dict_ = {}
    #     for j in range(topk_v.shape[1]):
    #         if topk_v[i][j] >= threshold:
    #             new_cluster.append(topk_i[i][j])
    #             scores += topk_v[i][j]
    #         else:
    #             break
    #     if len(new_cluster) >= min_community_size:
    #         dict_['new_cluster']=new_cluster
    #         dict_['scores']=scores
    #         extracted_communities.append(dict_)
    #
    # # Largest cluster first
    # extracted_communities = sorted(extracted_communities, key=lambda x: (len(x['new_cluster']), x['scores']), reverse=True)
    #
    # # Largest cluster first
    # extracted_communities = sorted(extracted_communities, key=lambda x: (len(x['new_cluster']), x['scores']),
    #                                reverse=True)

    return extracted_communities

def store_extract_one_cluster_data(file_index):

    topk_v, topk_i = read_one_cluster_data(dir, file_index, device)

    # topk_i = topk_i.type(torch.int32)

    extracted = extract_one_cluster_data(topk_v, topk_i, device)#[[id0,id1,id2,...],...]

    with open(f'{outputdir}/extracted_{file_index}.pkl',"wb") as fOut:
        pickle.dump({'extracted': extracted}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

for i in tqdm(range(start,end)):
    store_extract_one_cluster_data(i)