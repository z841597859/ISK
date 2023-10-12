# BCC: Batch_or_Cluster_or_Concept

# for unknown generator
def try_next_with_func(iter_obj, func):
    while True:
        try:
            e = next(iter_obj)
            func(e) # may can globally change some variables
        except StopIteration:
            break

# for generating data formed by BCCs
class batch_or_cluster_or_concept_distributor():
    def __init__(self, bcc_generator):
        self.bcc_source = bcc_generator
        self.distribute_log = []
    def __iter__(self):
        log = []
        while True:
            try:
                bcc = next(self.bcc_source)
                count_one_bcc = 0
                while True:
                    try:
                        yield next(bcc)
                        count_one_bcc += 1
                    except StopIteration:
                        log.append(count_one_bcc)
                        break
            except StopIteration:
                break
        self.distribute_log = log
    def custom_iter_for_turks(self):
        log = []
        while True:
            try:
                bcc = iter(next(self.bcc_source)[1]) #当做可忽略内存占用的最小单元,已经读取了一行数据了
                count_one_bcc = 0
                while True:
                    try:
                        yield next(bcc)
                        count_one_bcc += 1
                    except StopIteration:
                        log.append(count_one_bcc)
                        break
            except StopIteration:
                break
        self.distribute_log = log
    def custom_get_log_table_for_turks(self):
        log = []
        while True:
            try:
                bcc = next(self.bcc_source)[1]
                log.append(len(bcc))
            except StopIteration:
                break
        return log
    def get_log_table(self):
        log = []
        while True:
            try:
                bcc = next(self.bcc_source)
                count_one_bcc = 0
                while True:
                    try:
                        next(bcc)
                        count_one_bcc += 1
                    except StopIteration:
                        log.append(count_one_bcc)
                        break
            except StopIteration:
                break
        return log

# for generating a BCC one by one within the whole data
class Local_Bcc_Generator():
    def __init__(self, Whole_raw_generator, Bcc_length):
        self.whole_generator = Whole_raw_generator
        self.length = Bcc_length
    def __iter__(self):
        for i in range(self.length):
            try:
                yield next(self.whole_generator)
            except StopIteration:
                break

# for forming result like the shape of data of BCCs
class batch_or_cluster_or_concept_allocator():
    def __init__(self, result_generator, distribute_log):
        self.result_source = result_generator
        self.allocate_table = distribute_log
    def __iter__(self):
        for i in self.allocate_table:
            yield Local_Bcc_Generator(self.result_source, i)
