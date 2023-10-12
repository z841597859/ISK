from nltk import sent_tokenize
import re


def text_nltk_sents(text):#分句
    sents = re.sub(r'\.', '. ', text)
    sents = re.sub(r'\s+', ' ', sents)
    sents = sent_tokenize(sents)
    return sents


def Read_c4_text(line):
    text = line['text']
    return text

def Read_c4_text_to_sents(line):
    text = line['text']
    sents = iter(text_nltk_sents(text))
    return sents

def Read_txts_line(line):
    x = line.strip()
    return x

def Read_refer_turks_line(line):
    y = line[0]
    z = line[1:]
    return y, z

def Read_ours_csv_line(line):
    x = line[0]
    return x


def Read_ascent_source_sentences(line):
    x = iter([d['text'] for d in line['source_sentences']])
    return x

def Read_ascentpp_spo_template(line):
    # x = line['subject'] + ' ' + line['predicate'] + ' ' + line['object']
    x = line[5] + ' ' + line[6] + ' ' + line[7]
    return x

def Read_GenericsKB_GENERIC_SENTENCE(line):
    x = line[3]
    return x

def Read_quasimodo_first_sentences_source(line):
    x = line[6]
    x = x.split(' // ')
    x = [y.split(' x#x')[0] for y in x]
    x = x[0]
    return x

def Read_quasimodo_all_sentences_source(line):
    x = line[6]
    x = x.split(' // ')
    x = iter([y.split(' x#x')[0] for y in x])
    return x

def Read_aristo_Sentence(line):
    x = line[5]
    return x


def Read_top1000_concepts(line):
    line = line.strip()
    x = line.split('%')[0]
    y = x.split('[')[1]# txt文件最后不要有空行
    return y