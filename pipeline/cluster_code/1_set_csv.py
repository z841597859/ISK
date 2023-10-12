import pandas as pd
import csv
from tqdm import tqdm

inputdir = '/home/zzw/_test37_/Experiments/predict_raw_k'
outputdir = '/home/zzw/_test37_/Experiments/set_raw_k_with_i'

for fileindex in tqdm(range(140,256)):

    df = pd.read_csv(f'{inputdir}/predict_{fileindex}/emb_topredict_{fileindex}.csv')
    text = df['text'].to_list()
    id = df['id'].to_list()

    # #遍历方法-元组in-03:36<32:32, 216.98s/it
    # rows = zip(text,id)
    # set_rows =[]
    # for row in rows:
    #     row_=list(row)
    #     if row_ not in set_rows:
    #         set_rows.append(row_)

    # # 遍历方法-先sent表再id表-too slow
    # set_text = []
    # set_id = []
    # for i,sent in enumerate(text):
    #     mark = True
    #     for j,sent_ in enumerate(set_text):
    #         if sent == sent_:
    #             if id[i] == set_id[j]:
    #                 mark = False
    #                 break
    #     if mark == True:
    #         set_text.append(sent)
    #         set_id.append(id[i])
    # set_rows=[[set_text[i],set_id[i]] for i in range(len(set_text))]

    # 取巧方法（id增序）
    set_text = []
    set_id = []
    last_id = -1
    set_text_= []
    set_id_ = []
    for i,id_ in enumerate(id):
        if id_ != last_id:
            set_text += set_text_
            set_id += set_id_
            set_text_ = []
            set_id_ = []
            set_text_.append(text[i])
            set_id_.append(id[i])
            last_id = id_
        else:
            if text[i] not in set_text_:
                set_text_.append(text[i])
                set_id_.append(id[i])
                if i == len(id)-1:
                    set_text += set_text_
                    set_id += set_id_
            else:
                pass
    set_rows = [[set_text[i], set_id[i]] for i in range(len(set_text))]
    print(len(set_rows))

    with open(f'{outputdir}/set_raw_knowledge_{fileindex}.csv', 'w')as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'id'])
        writer.writerows(set_rows)

