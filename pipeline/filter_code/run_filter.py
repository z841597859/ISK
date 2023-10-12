from multiprocessing import Pool
from c4_reader import read_one_file
from filter import main_filter_text_dict
from tqdm import tqdm
import gzip
import json
from pathlib import Path

start_file_index=948
end_file_index=950
NUM_PROCESSORS=8
dir = '/D/zzw-c4/c4/en'
outputdir = '/D/zzw-c4/c4-output'

# index_to_reason=['',#False
#                 'len_of_str>200',#1
#                 '出现少于15次英文字母',#2
#                 '感叹句和问句',#3
#                 '性别三人称',#4
#                 '句首指示代词_i_me',#5
#                 '一二人称共现',#6
#                 'Include_Bad_Signals',#7
#                 'len_of_tok<=3or>=40',#8
#                 'NE-BW-pos-propn',#9
#                 'NE-BW-ent_type_',#10
#                 'NE-BW-tag-NFPADD',#11
#                 'NE-BW-dep-ROOT',#12
#                 'NE-BW-N0NBef-they',#13
#                 '整个句子提取不出一组spo',#14
#                 '所有spo都缺少主语和宾语']#15
# reasons_len = len(index_to_reason)

def run_filter(file_index):

    filename = f'{dir}/c4-train.{file_index:05d}-of-01024.json.gz'

    document = read_one_file(filename)  # list of {'text':---,'timestamp':---,'url':---}

    with Pool(NUM_PROCESSORS) as p:
        result = list(tqdm(iterable=(p.imap(main_filter_text_dict, document)), desc='Processing'))

    outputfilename=f'{outputdir}/filteroutput-{file_index:05d}-of-01024.json.gz'

    count_text=0
    count_sent=0
    count_filteroutput=0

    with gzip.open(outputfilename, 'wt', encoding='utf-8') as f:
        for dict in result:

            dict_ = {}

            if dict['pass'] != []:

                dict_['id'] = count_text
                dict_['text'] = dict['text']
                dict_['filteroutput'] = dict['pass']

                count_text+=1
                count_sent+=len(dict['pass'])+len(dict['filted'])
                count_filteroutput+=len(dict['pass'])

            else:

                count_sent+=len(dict['filted'])

            if dict_ != {}:

                f.write(json.dumps(dict_))
                f.write('\n')

        print('text included output:',count_text)
        print('total sents num is:',count_sent)
        print('filteroutput sents num is:',count_filteroutput)

    # for i in range(1,reasons_len):

    #     count_2=0

    #     output_dir = Path(f'{outputdir}/{index_to_reason[i]}')
    #     output_dir.mkdir(exist_ok=True)
    #     output_file = output_dir / Path(f"{index_to_reason[i]}-filtered.json.gz")

    #     with gzip.open(output_file, 'wt', encoding='utf-8') as f:

    #         for dict in result:

    #             for filted_tuple in dict['filted']:

    #                 dict_ = {}

    #                 if filted_tuple[1] == i:

    #                     dict_['s'] = filted_tuple[0]

    #                     f.write(json.dumps(dict_))
    #                     f.write('\n')

    #                     count_2+=1

    #         print(index_to_reason[i],count_2)

    # with gzip.open(outputfilename, 'rt', encoding='utf-8') as f:
    #     count=0
    #     for line in f:
    #         sents = json.loads(line)['pass']
    #         for sent in sents:
    #             print(sent)
    #             print('\n')
    #             count+=1
    #     print(count)

def main():

    for i in range(start_file_index, end_file_index):
        
        print(f'---ready to filt {i:05d}-of-01024---')

        run_filter(i)

        print(f'---finished writing {i:05d}-of-01024---')

    return

main()