from tqdm import tqdm
import csv

class Read_tsv_no_header():
    def __init__(self, data_path, line_func):
        self.data_path = data_path
        self.line_func = line_func
    def __iter__(self):
        with open(self.data_path) as f:
            for line in f:
                line = line.strip().split("\t")
                x = self.line_func(line)
                yield x

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

class Read_txt():
    def __init__(self, data_path, line_func):
        self.data_path = data_path
        self.line_func = line_func
    def __iter__(self):
        with open(self.data_path) as f:
            for line in f:
                x = self.line_func(line)
                yield x

def Read_top1000_concepts(line):
    line = line.strip()
    x = line.split('%')[0]
    y = x.split('[')[1]# txt文件最后不要有空行
    return y

# refer_dir='/E/zzw/KbData/ReferData/turks_data_csk.tsv'
# refer = iter(Read_tsv_no_header(refer_dir, lambda x:x))
#['surgeon', 'Surgeons are the ones who operate', 
# 'Surgeons are medical specialists', 'Surgeons work in the operating rooms', 'Surgeons have a good salary']
#['surgeon', 'Surgeons are precise.', 'Surgeons have studied.', 
# 'Surgeons work in hopsitals.', 'Surgeons work under sterile conditions.']
#600

# refer_dir='/E/zzw/KbData/ReferData/related_words.tsv'
# refer = iter(Read_tsv_no_header(refer_dir, lambda x:x))
#['surgeon', 'peoples', 'and', 'is', 'many', 'lives', 
# 'hands', 'with', 'cure', 'ones', 'save', 'heroes', 'works', 'precise', ...]
#['moth', 'wooden', 'and', 'proliferate', 'various', 'causes', 'is', 'lepidoptera',
#  'caterpillars', 'group', 'they', 'shapes', 'moth', 'serious', ...]
#100

# refer_dir='/E/zzw/KbData/ReferData/ascent_subjects.csv'
# refer = iter(Read_csv(refer_dir, lambda x:x[2]))
#'super_subject'
#'aa'
#64780

refer_dir='/E/zzw/KbData/ReferData/top5000.txt'
refer = iter(Read_txt(refer_dir, Read_top1000_concepts))
#'time'
#'year'
#5000


# for unknown generator
def try_next_with_func(iter_obj, func):
    while True:
        try:
            e = next(iter_obj)
            func(e) # may can globally change some variables
        except StopIteration:
            break

a = next(refer)
b = next(refer)

Count=2
for i in tqdm(refer):
    Count+=1

print('')