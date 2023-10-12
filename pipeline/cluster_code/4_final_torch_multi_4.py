from tqdm import tqdm
from torch.multiprocessing import Pool, Lock
from for_final_tool_4 import main_cluster,read_strat_index_list
from functools import partial
import torch

start_file_index = 428
end_file_index = 429

Whole_file_num =1024
Batch_size = 10240

Topk = 200

dir = '/media/c/D/zzw-c4/tensor_fc_with_i'
outputdir = '/media/c/D/zzw-c4/cluster_data'

# Gpu_list=['cuda:1','cuda:2','cuda:3','cuda:4','cuda:5','cuda:6']
Gpu_list=['cuda:4','cuda:5','cuda:6']

if __name__ == "__main__":

    p = Pool(
        len(Gpu_list),
        initializer=tqdm.set_lock,
        initargs=(Lock(),),
    )

    Strat_index_list = read_strat_index_list(dir,Whole_file_num)

    para_main_cluster = partial(main_cluster,
                                topk=Topk,batch_size=Batch_size,whole_file_num=Whole_file_num,gpu_list=Gpu_list,
                                dir=dir,outputdir=outputdir,strat_index_list=Strat_index_list,tqdm_base=start_file_index)

    file_list = [i for i in range(start_file_index, end_file_index)]

    num_file_batch = int(len(file_list)/len(Gpu_list)) + 1
    if len(file_list)%len(Gpu_list) == 0:
        num_file_batch -= 1

    for i in range(0,num_file_batch):

        start= i * len(Gpu_list)

        file_list_batch = file_list[start : start+len(Gpu_list)]

        list(p.imap(para_main_cluster, file_list_batch))

        torch.cuda.empty_cache()  # 释放显存

        # list(p.imap(para_main_cluster, file_list))

        # p.imap(main_cluster, file_list)
        # list(tqdm(iterable=(p.imap_unordered(main_cluster, file_list)), desc='Processing'))
        # with Pool(Num_gpus) as p:
        #     list(tqdm(iterable=(p.imap_unordered(for_one_gpu, file_list)), desc='Processing'))
        # list(tqdm(iterable=(p.imap(para_main_cluster, file_list)), desc='Processing'))

    p.close()
    p.join()


