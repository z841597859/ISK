from sentence_transformers import SentenceTransformer, models, LoggingHandler
import os
import logging
import joblib
from tqdm import tqdm
from multiprocessing import Pool
from utils.Reader_s import *
from utils.KBs_line_func import *
from utils.BCC_iter import batch_or_cluster_or_concept_distributor
from utils.Writer_s import txt_write_one_row

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

NUM_PROCESSORS = 8
clf = joblib.load('/home/zzw/_test37_/6.1/models/10_0.1_distilroberta-base_svc-2.22-filted-balanced.svc')
outputfolder = '/E/zzw/OutputData/512-1025'
os.makedirs(outputfolder, exist_ok=True)

start = 514
end = 1024
inputfolder ='/E/zzw/OutputData/tmp-512-1024'

def single_for_ndarray(emb_row):
    pred = clf.predict([emb_row])
    return pred

class Read_csv():
    def __init__(self, data_path, line_func):
        self.data_path = data_path
        self.line_func = line_func
    def __iter__(self):
        with open(self.data_path) as f:
            csvReader = csv.reader(f)
            for line in csvReader:
                x = self.line_func(line)
                yield x

# Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':

    # # Define the model
    word_embedding_model = models.Transformer('distilroberta-base', max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool(target_devices=['cuda:1'])#['cuda:1', 'cuda:2', 'cuda:3','cuda:4','cuda:6']

    for file_index in range(start,end):

        inputfilename=f'toclassify-{file_index:05d}-of-01024.csv'
        inputdir = f'{inputfolder}/{inputfilename}'

        # Create a large list of 100k sentences
        filtered_sents=iter(Read_csv(inputdir, lambda x:x[0]))
        next(filtered_sents)
        # filtered_sents_flatten=iter(batch_or_cluster_or_concept_distributor(filtered_sents))
        sentences = list(filtered_sents)

        # Compute the embeddings using the multi-process pool
        emb = model.encode_multi_process(sentences, pool)
        print("Embeddings computed. Shape:", emb.shape)

        with Pool(NUM_PROCESSORS) as p:
            result_generator = p.imap(single_for_ndarray, emb)

            pred = []
            for result in tqdm(result_generator):
                pred.append(result[0])

        outputfilename=f'c4-train.{file_index:05d}-of-01024_classifyoutput.txt'
        outputdir = f'{outputfolder}/{outputfilename}'
        with open(outputdir,'w') as fOut:
            for i,p in enumerate(pred):
                if p == 1:
                    txt_write_one_row(sentences[i], fOut)
        
    # Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)
