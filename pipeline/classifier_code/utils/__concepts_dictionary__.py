from Reader_s import *
from KBs_line_func import *

concepts_folder = '/E/zzw/data/for_evaluation'

Concepts_150_dir=f'{concepts_folder}/150-subjects.tsv'
Concepts_150 = iter(Read_tsv_header(Concepts_150_dir, lambda x:x[1]))
title = "domain: animal, occupation, engineering"
a = next(Concepts_150)

ascent_subjects_dir=f'{concepts_folder}/ascent_subjects.csv'
ascent_subjects=iter(Read_csv(ascent_subjects_dir, lambda x:x[2]))
title = "['subject', 'type', 'super_subject']"
next(ascent_subjects)
b = next(ascent_subjects)

quasimodo_subjects_dir = f'{concepts_folder}/subjects.txt'
quasimodo_subjects=iter(Read_txt(quasimodo_subjects_dir, lambda x:x.strip()))
c = next(quasimodo_subjects)

top1000_dir=f'{concepts_folder}/top1000.txt'
top1000=iter(Read_txt(top1000_dir, Read_top1000_concepts))
d = next(top1000)

top5000_dir=f'{concepts_folder}/top5000.txt'
top5000=iter(Read_txt(top5000_dir, Read_top1000_concepts))
e = next(top5000)

print('')