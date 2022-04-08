import pickle
import random
import numpy as np
import os

base_dir = 'data/ontonotes'
few_ind = 2

def get_format_and_stats(fn):
    with open(fn) as fin:
        lines = fin.readlines()
    counts = {}
    res = []
    sent = []
    for line in lines:
        if line.strip() == '':
            res.append(sent)
            sent = []
            continue
        line = line.strip().split()
        if line[-1].startswith('B'):
            counts[line[-1]] = counts.get(line[-1], 0) + 1
        sent.append(line)
    mean_count = np.mean(list(counts.values()))
    return res, mean_count

def write_format(sents, fn):
    print(sents[:5])
    with open(fn, 'w') as fout:
        for sent in sents:
            for line in sent:
                fout.write('\t'.join(line) + '\n')
            fout.write('\n')

if __name__ == '__main__':
    base_data, base_count  = get_format_and_stats(os.path.join(base_dir, 'train_base.txt'))
    few_data, few_count = get_format_and_stats(os.path.join(base_dir, 'train_few%d.txt'%few_ind))
    num_repeat = min(base_count // few_count, 100)
    print(num_repeat)
    res = base_data + (few_data * num_repeat)
    random.shuffle(res)
    write_format(res, os.path.join(base_dir, 'concat_train.txt'))




