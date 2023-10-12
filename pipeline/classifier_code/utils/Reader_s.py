import pickle
import csv
import json
import gzip
import pandas as pd

# All should be used with try and except StopIteration

class Read_txt():
    def __init__(self, data_path, line_func):
        self.data_path = data_path
        self.line_func = line_func
    def __iter__(self):
        with open(self.data_path) as f:
            for line in f:
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

class Read_csv_header():
    def __init__(self, data_path, line_func):
        self.data_path = data_path
        self.line_func = line_func
    def __iter__(self):
        with open(self.data_path) as f:
            csvReader = csv.reader(f)
            next(csvReader)
            for line in csvReader:
                x = self.line_func(line)
                yield x

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

class Read_tsv_header():
    def __init__(self, data_path, line_func):
        self.data_path = data_path
        self.line_func = line_func
    def __iter__(self):
        with open(self.data_path) as f:
            tsvReader = csv.reader(f, delimiter='\t')
            next(tsvReader)
            for line in tsvReader:
                x = self.line_func(line)
                yield x

class Read_pickle(): # EOFError: Ran out of input
    def __init__(self, data_path ,line_func):
        self.data_path = data_path
        self.line_func = line_func
    def __iter__(self):
        with open(self.data_path, 'rb') as fIn: # can't "for line in fIn:"
            while True:
                try:
                    line = pickle.load(fIn)
                    x = self.line_func(line)
                    yield x
                except EOFError:
                    break

class Read_json_gz():
    def __init__(self, data_path ,line_func):
        self.data_path = data_path
        self.line_func = line_func
    def __iter__(self):
        with gzip.open(self.data_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                x = self.line_func(line)
                yield x

class Read_jsonl():
    def __init__(self, data_path, line_func):
        self.data_path = data_path
        self.line_func = line_func
    def __iter__(self):
        with open(self.data_path) as f:
            for line in f:
                line = json.loads(line)
                x = self.line_func(line)
                yield x


def read_strat_index_list(dir, Whole_file_num):
    df = pd.read_csv(f'{dir}/index_for_set_raw_knowledge_{Whole_file_num}.csv')
    start_index_list = df['start_index'].to_list()
    return start_index_list
