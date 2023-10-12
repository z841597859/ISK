import pickle
import os

# with open(dir, "wb") as fOut:
def pickle_write_one_row(row, fOut):
    pickle.dump(row, fOut)

def pickle_write_whole(rows, result_name, dir):
    assert os.path.exists(dir) == False
    with open(dir, "wb") as fOut:
        pickle.dump({f'{result_name}': rows}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_read_whole(result_name, dir):
    with open(dir, "rb") as fIn:
        stored_data = pickle.load(fIn)[result_name]
    return stored_data

# with open(dir, "w") as f:
def txt_write_one_row(row, f):
    f.write(row)
    f.write('\n')

def pro_batch_working(raw_results, batchsize, working_func):
    batch_count=0
    while True:
        try:
            working_func(raw_results, batchsize, batch_count)
            batch_count += 1
        except StopIteration:
            break