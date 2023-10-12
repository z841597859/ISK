from tqdm import trange
import torch
import pickle

def read_tensor_pkl(dir, file_index, device):
    with open(f'{dir}-{file_index}', "rb") as fIn:
        stored_data = pickle.load(fIn) # cpu, float16
        file_embs = stored_data['embeddings'].to(device)
    return file_embs

def get_tensor_emb(emb_list, open_file_num, device):
    open_emb = emb_list[open_file_num].to(device)
    return open_emb

def batch_open_cluster(batch_embs,open_file_num,device,topk,emb_list):

    open_embs = get_tensor_emb(emb_list, open_file_num, device)

    # Compute cosine similarity scores
    cos_scores = torch.mm(batch_embs, open_embs.transpose(0, 1))

    top_k_values, top_k_ids = cos_scores.topk(k=topk)

    top_k_ids = top_k_ids.type(torch.int32)

    return top_k_values, top_k_ids

def merge_topk_results(tkv_0,tki_0,tkv_1,tki_1,device,topk):

    tkv_0 = torch.cat((tkv_0, tkv_1), dim=1)
    tki_0 = torch.cat((tki_0, tki_1), dim=1)

    tkv, trans_tki = tkv_0.topk(k=topk, largest=True)

    tki = torch.empty((0, topk), dtype=torch.int32, device=device)

    for i in range(0, len(tkv)):
        tki_row = tki_0[i, trans_tki[i]].unsqueeze(0)
        tki = torch.cat((tki, tki_row), dim=0)

    return tkv, tki

def batch_cluster(batch_embs, whole_file_num, row_len, device, topk, emb_list, strat_index_list):

    batch_whole_k_values = torch.empty((row_len, 0), dtype=torch.float16, device=device)
    batch_whole_k_ids = torch.empty((row_len, 0), dtype=torch.int32, device=device)

    for open_file_num in range(0, whole_file_num):

        topk_values, topk_ids = batch_open_cluster(
            batch_embs,open_file_num,device,topk,emb_list)

        topk_ids += strat_index_list[open_file_num]

        batch_whole_k_values, batch_whole_k_ids = merge_topk_results(
            batch_whole_k_values, batch_whole_k_ids, topk_values, topk_ids,
            device, topk)

    return batch_whole_k_values, batch_whole_k_ids

def main_cluster(file_index, emb_list, topk, batch_size, whole_file_num, 
                 gpu_list, inputfilename, outputfilename, strat_index_list, tqdm_base):

    num_gpus = len(gpu_list)
    gpu_id = file_index % num_gpus
    device = gpu_list[gpu_id]
    print(f'start clustering {file_index} by {device}')

    file_embs = read_tensor_pkl(inputfilename, file_index, device)

    whole_k_values = torch.empty((0, topk), device=device)  #dtype=torch.float16,
    whole_k_ids = torch.empty((0, topk), dtype=torch.int32, device=device)

    batch_num = int(len(file_embs)/batch_size)+1
    if len(file_embs) % batch_size == 0:
        batch_num -= 1
    
    t = trange(batch_num, desc=f'Processing{file_index}by{device}', position=file_index-tqdm_base)
    for i in t:
        start_idx = i * batch_size
        row_len = min(batch_size, len(file_embs) - start_idx) #for concat by dim 1
        
        batch_whole_k_values, batch_whole_k_ids = batch_cluster(file_embs[start_idx:start_idx+row_len], whole_file_num,
                                                                row_len, device, topk, emb_list, strat_index_list)
        
        whole_k_values = torch.cat((whole_k_values, batch_whole_k_values), dim=0)
        whole_k_ids = torch.cat((whole_k_ids, batch_whole_k_ids), dim=0)

    with open(f'{outputfilename}-{file_index}',"wb") as fOut:
        pickle.dump({'k_values': whole_k_values.to('cpu'), 'k_ids': whole_k_ids.to('cpu')},
                    fOut, protocol=pickle.HIGHEST_PROTOCOL)

    del file_embs
    del whole_k_values
    del whole_k_ids
    del batch_whole_k_values
    del batch_whole_k_ids
    torch.cuda.empty_cache()  # 释放显存

    return None
