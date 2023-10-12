from Reader_s import *
from KBs_line_func import *
from BCC_iter import batch_or_cluster_or_concept_distributor
from Eval_assos import read_subjects

# sent-like KBs:
GenericsKB_dir='/E/zzw/data/KBs/Sent_KB/GenericsKB/GenericsKB.tsv'
GenericsKB = Read_tsv_header(GenericsKB_dir, Read_GenericsKB_GENERIC_SENTENCE)

GenericsKB_Best_dir='/E/zzw/data/KBs/Sent_KB/GenericsKB/GenericsKB-Best.tsv'
GenericsKB_Best = Read_tsv_header(GenericsKB_Best_dir, Read_GenericsKB_GENERIC_SENTENCE)

ours_folder = '/media/c/D/zzw-c4/set_raw_k_with_i'
for file_index in range(0,512):
        ours_dir = f'{ours_folder}/set_raw_knowledge_{file_index}.csv'
        ours = iter(Read_csv(ours_dir, Read_ours_csv_line))
        next(ours) # without header

# tuple-like KBs:
aristo_tuple_kb_dir='/E/zzw/data/KBs/Tuple_KB/Aristo_tupleKB/TACL-kb.tsv'
quasimodo_dir='/E/zzw/data/KBs/Tuple_KB/quasimodo/quasimodo43.tsv'
ascent_dir='/E/zzw/data/KBs/Tuple_KB/ascent/ascent-v1.0.0.json.gz'

ascentpp_dir='/E/zzw/data/KBs/Tuple_KB/ascentpp/ascentpp.csv'
ascentpp = iter(Read_csv(ascentpp_dir, lambda x:x))

# Source Sents from tuple-like KBs:
aristo_tuple_kb = iter(Read_tsv_header(aristo_tuple_kb_dir, Read_aristo_Sentence))

quasimodo_all_sentences_source = iter(Read_tsv_no_header(quasimodo_dir, Read_quasimodo_all_sentences_source))
quasimodo_all_sentences_source_flatten = iter(batch_or_cluster_or_concept_distributor(quasimodo_all_sentences_source))

quasimodo_first_sentences_source = iter(Read_tsv_no_header(quasimodo_dir, Read_quasimodo_first_sentences_source))

ascent_source_sentences = iter(Read_json_gz(ascent_dir, Read_ascent_source_sentences))
ascent_source_sentences_flatten = iter(batch_or_cluster_or_concept_distributor(ascent_source_sentences))

# Template Sents from tuple-like KBs:
ascentpp_spo_template = iter(Read_csv(ascentpp_dir, Read_ascentpp_spo_template))

# Knowledge Intensive Corpus:
aristominicorpus_dir = '/E/zzw/data/KBs/KI_corpus/aristo-mini-corpus/aristo-mini-corpus-v1.txt'
aristominicorpus = iter(Read_txt(aristominicorpus_dir, Read_txts_line))

ARC_Corpus_dir = '/E/zzw/data/KBs/KI_corpus/ARC-V1-Feb2018/ARC_Corpus.txt'
ARC_Corpus = iter(Read_txt(ARC_Corpus_dir, Read_txts_line))

QASC_Corpus_dir = '/E/zzw/data/KBs/KI_corpus/QASC_Corpus/QASC_Corpus.txt'
QASC_Corpus = iter(Read_txt(QASC_Corpus_dir, Read_txts_line))


# for eval recall
subjects_dir = '/E/zzw/data/for_evaluation/subjects.txt'
subjects = read_subjects(subjects_dir)

refer_dir="/E/zzw/data/for_evaluation/turks_data_csk.tsv"
refer_line_generator = iter(Read_tsv_no_header(refer_dir, lambda x:x))

quasimodo_line_generator = iter(Read_tsv_no_header(quasimodo_dir, lambda x:x))

related_words_dir ="/E/zzw/data/for_evaluation/related_words.tsv"
related_words_generator = iter(Read_tsv_no_header(related_words_dir, lambda x:x))


oie_result_folder = '/E/zzw/outputdata/oie_result'
oie_file_name = 'aristo-mini-corpus-v1_oie_result.pkl'
oie_result_dir = f'{oie_result_folder}/{oie_file_name}'
oie_result = iter(Read_pickle(oie_result_dir, lambda x:iter(x)))
oie_result_flatten = iter(batch_or_cluster_or_concept_distributor(oie_result))


QAtxt_dir= '/E/zzw/outputdata/for_QA_txts/waterloo-zzw-bare.txt'
QAtxt_generator=iter(Read_txt(QAtxt_dir, Read_txts_line))
