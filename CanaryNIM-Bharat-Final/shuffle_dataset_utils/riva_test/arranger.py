import json
import random
import logging
import pathlib


VALID_MODES=['random', 'best', 'worst']
BATCH_SIZE=1
SEED=None
PAD=False
logger = logging.getLogger(pathlib.Path(__file__).name)


class Arranger:
    def __init__(self, seed=SEED, sampled_runs=1, sorted_run=1, pad=PAD):

        self.seed = seed
        self.sampled_runs = sampled_runs
        self.sorted_run = sorted_run
        if seed is not None:
            if isinstance(seed, list):
                self.sampled_runs=len(seed)

        self.pad = pad


    @staticmethod
    def __validate_bs(bs):
        if not (bs.bit_count()==1):
            raise ValueError(f"Batch size '{bs}' is not valid. Should be one a power of 2")


    def shuffle(self, manifest, skip_sort=False, batch_size=BATCH_SIZE):
        self.__validate_bs(batch_size)

        if isinstance(manifest, list):
            input_mf = manifest
        else:
            input_mf = []
            with open(manifest,'r') as mfile:
                for line in mfile:
                    input_mf.append(json.loads(line))

        if  skip_sort:
            sorted_mf=input_mf
        else:
            sorted_mf = sorted(input_mf, key=lambda d: d['duration'], reverse=True)

        pad = len(sorted_mf)-len(sorted_mf)//batch_size*batch_size
        if pad > 0:
            sorted_mf.extend(sorted_mf[pad-batch_size:])

        # return best, worst, sampled
        return self.best(sorted_mf), self.worst(sorted_mf, batch_size=batch_size), self.sampled(sorted_mf, self.seed)


    def best(self, sorted_mf):

        return sorted_mf * self.sorted_run


    def worst(self, sorted_mf, batch_size=BATCH_SIZE):
        self.__validate_bs(batch_size)
        worst_list=[]
        no_lists=len(sorted_mf)//batch_size

        for batch in range(0,no_lists):
            worst_list.extend([sorted_mf[i] for i in range(batch,len(sorted_mf),no_lists)])

        return worst_list*self.sorted_run


    def sampled(self, sorted_mf, seeds):
        if not isinstance(seeds, list):
            if seeds is not None:
                seeds = [seeds]

        sampled_mf = []
        if seeds is not None:
            n_runs=self.sampled_runs/len(seeds)
            for seed in seeds:
                if seed is not None:
                    random.seed(seed)
                sampled_mf.extend(random.sample(sorted_mf, len(sorted_mf)) * n_runs)
        else:
            for i in range(0,self.sampled_runs):
                sampled_mf.extend(random.sample(sorted_mf, len(sorted_mf)))

        return sampled_mf
