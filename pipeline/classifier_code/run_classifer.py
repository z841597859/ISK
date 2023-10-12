from sentence_transformers import SentenceTransformer, models, LoggingHandler
import pandas as pd
import pickle
import os
import logging
import torch
import joblib
from tqdm import tqdm
import csv

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

clf=joblib.load('/home/zzw/_test37_/__pipeline__/classifier_code/__classifier-data__/3.10/10_0.1_distilroberta-base_svc-2.22-filted-balanced.svc')

start_file_index=512
end_file_index=1024

inputdir = '/E/zzw/OutputData/tmp-512-1024'

# Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':

    # # Define the model
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    word_embedding_model = models.Transformer('distilroberta-base', max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool(target_devices=['cuda:0', 'cuda:1'])#['cuda:1', 'cuda:2', 'cuda:3','cuda:4','cuda:6']

    for file_index in tqdm(range(start_file_index, end_file_index)):

        print(f'---ready to classify {file_index:05d}-of-01024---')

        inputfile = f'toclassify-{file_index:05d}-of-01024.csv'

        df = pd.read_csv(f'{inputdir}/{inputfile}')
        text = df['text'].to_list()
        id = df['id'].to_list()

        outputdir = '/E/zzw/OutputData/512-1025'
        output_folder = f"{outputdir}/predict_{file_index}"
        os.makedirs(output_folder, exist_ok=True)

        # Create a large list of 100k sentences
        sentences = text

        # Compute the embeddings using the multi-process pool
        emb = model.encode_multi_process(sentences, pool)#sentences, pool
        print("Embeddings computed. Shape:", emb.shape)
        # print(torch.cuda.current_device())

        p = clf.predict(emb)

        count=0
        rows=[]
        for index,pred in enumerate(p):
            if pred == 1:
                row=[]
                count+=1
                row.append(sentences[index])
                row.append(id[index])
                rows.append(row)
        print(count)
        with open(f'{output_folder}/emb_topredict_{file_index}.csv','w')as f:
            writer=csv.writer(f)
            writer.writerow(['text','id'])
            writer.writerows(rows)

        print(f'---finished classify {file_index:05d}-of-01024---')

    # Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)