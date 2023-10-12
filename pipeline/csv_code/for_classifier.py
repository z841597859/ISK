import gzip
import json
import csv

start=514
end=1024
inputdir = '/D/zzw-c4/c4-output'
outputdir="/E/zzw/OutputData/tmp-512-1024"

def processing(sent):

    sent_ = sent.lower().lstrip().rstrip().replace('\0', '')
    
    return sent_

def to_csv_for_classifier(file_index):

    inputfilename=f'{inputdir}/filteroutput-{file_index:05d}-of-01024.json.gz'
    rows=[]

    with gzip.open(inputfilename, 'rt', encoding='utf-8') as f:

        for line in f:

            dict= json.loads(line)

            for sent in dict['filteroutput']:

                row=[]
                row.append(processing(sent))
                row.append(dict['id'])
                rows.append(row)

    print('rows len is:',len(rows))
    
    with open(f'{outputdir}/toclassify-{file_index:05d}-of-01024.csv', 'wt', encoding='utf-8') as f:
        f_writer=csv.writer(f)
        f_writer.writerow(['text','id'])
        f_writer.writerows(rows)

    return


def main():

    for i in range(start,end):

        to_csv_for_classifier(i)


main()

