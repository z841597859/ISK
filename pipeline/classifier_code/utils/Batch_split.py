# BCC: Batch_or_Cluster_or_Concept
from utils.BCC_iter import try_next_with_func
from functools import partial

# for forming result with the shape of batchsize
class Bcc():
    def __init__(self, Whole_raw_generator, size):
        self.whole_raw_generator = Whole_raw_generator
        self.size = size
        self.log = 0
    def __iter__(self):
        for i in range(self.size):
            try:
                yield next(self.whole_raw_generator)
                self.log += 1
            except StopIteration: # whole_raw_generator StopIteration, this Bcc also end
                break

# for creating a distributor of Bccs
class Batch_distributor():
    def __init__(self, Whole_raw_generator, batch_size):
        self.whole_raw_generator = Whole_raw_generator
        self.batch_size = batch_size
        self.distribute_log = []
    def __iter__(self):
        while True:
            try:
                yield Bcc(self.whole_raw_generator, self.batch_size)
                self.distribute_log.append(self.batch_size)
            except StopIteration:
                print('something error')

# for final batch working
def batch_working(raw_generator, batchsize, working_func):

    b_d = Batch_distributor(raw_generator, batchsize)
    b_d_g = iter(b_d)

    last_log = batchsize
    batch_index = 0

    while last_log == batchsize:
        bcc = next(b_d_g)
        batch = iter(bcc)

        #working_func should be a func with 'batch_index' parameter
        # para = partial(working_func, batch_index= batch_index)
        # try_next_with_func(batch, para)

        working_func(batch, batch_index = batch_index)

        # while True:
        #     try:
        #         data = next(batch)
        #         ###################
        #         working_func(data)
        #         ###################
        #     except StopIteration:  # Bcc StopIteration, this batch end
        #         break

        last_log = bcc.log
        batch_index+=1
    if last_log != 0:
        b_d.distribute_log.append(last_log)
    return b_d.distribute_log
# custom part for proprocessing
# def working_func(batch_result, batch_index):
#     outputfile = f'{outputfolder}/.....{batch_index}'
#     assert check_before_writing(outputfile) == True
#     with open(outputfile, "wb") as fOut:
#         for r in tqdm(batch_result):
#             if r != False:
#                 pickle_write_one_row(r, fOut)
# batch_working(result_generator, batchsize, working_func)
