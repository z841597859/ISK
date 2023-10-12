import torch
import pickle
import pandas as pd
from tqdm import tqdm

dir = '/media/c/D/zzw-c4/cluster_extracted'
outputdir = '/home/zzw/_test37_/Experiments/result_and_sentences'
whole_file_num = 512

def read_one_extracted_data(dir,file_index):
    with open(f'{dir}/extracted_{file_index}.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        extracted = stored_data['extracted']
    return extracted

def community_detection(extracted_communities, max_community_size=20, min_community_size=10):

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in tqdm(enumerate(extracted_communities)):#隐含：优先考虑（去重id前的）大簇//和高scores簇

        # community = sorted(community) #不会对extracted_communities修改
        non_overlapped_community = []

        for idx in community:

            if len(non_overlapped_community) >= max_community_size:
                break

            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)#簇元素数降序//，保持去重id前高scores顺序

    return unique_communities


def main():

    extracted_communities = []
    for file_index in range(0,whole_file_num):
        extracted_communities_ = read_one_extracted_data(dir, file_index)
        extracted_communities += extracted_communities_

    unique_communities = community_detection(extracted_communities)

    print(len(unique_communities))
    print(len(unique_communities[-1]))

    with open(f'{outputdir}/512-512-result-20-10.pkl', "wb") as fOut:
        pickle.dump({'unique_communities': unique_communities},
                    fOut, protocol=pickle.HIGHEST_PROTOCOL)


main()
