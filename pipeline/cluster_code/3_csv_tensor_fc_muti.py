from sentence_transformers import SentenceTransformer, util, LoggingHandler
import os
import csv
import time
import pickle
import torch
import logging
import pandas as pd
from tqdm import tqdm

inputdir='/home/zzw/_test37_/Experiments/set_raw_k_with_i'
outputdir='/media/c/D/zzw-c4/tensor_fc_with_i'

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':

    # Model for computing sentence embeddings. We use one trained for similar questions detection
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool(target_devices=['cuda:2', 'cuda:3', 'cuda:4', 'cuda:5'])

    for i in tqdm(range(0,512)):

        df = pd.read_csv(f'{inputdir}/set_raw_knowledge_{i}.csv')

        # Create a large list of 100k sentences
        sentences = df['text'].tolist()
        ids = df['id'].tolist()

        # corpus_sentences = list(corpus_sentences)
        emb = model.encode_multi_process(sentences, pool, batch_size=64)

        emb = torch.tensor(emb)

        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        #Store sentences & embeddings on disc
        with open(f'{outputdir}/{i}_tensor_fc.pkl', "wb") as fOut:
            pickle.dump({'embeddings': emb, 'text': sentences, 'id': ids}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    # Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)