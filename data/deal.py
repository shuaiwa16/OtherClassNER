import pickle
import random
import os
import numpy as np

few_words = {'conll' : ['PER'], 
             're3d' : ['Nationality', 'Weapon', 'Person'],
             'ontonotes' : ['PERSON', 'MONEY', 'PERCENT', 'LANGUAGE', 'NORP'],
             'wunut': ['product', 'person'],
             'cluener': ['name', 'government', 'scene', 'game'],
             'muc': ['phys_tgt_id', 'incident_instrument_id']}

def get_format(lines):
    res = []
    cr = []
    for line in lines:
        if line.strip() == '':
            res.append(cr)
            cr = []
            continue
        wl = line.strip().split()
        if len(wl) < 2:
            print(wl)
        assert len(wl) > 1
        cr.append(wl)
    return res

def get_labels(lines):
    res = []
    for line in lines:
        for w in line:
            res.append(w[-1].strip().split('-')[-1])
    res = list(set(res))
    print(res)
    return res
    
def get_class(lines, in_words, out_words):
    res = []
    for line in lines:
        in_flag = False
        out_flag = False
        for w in line:
            if w[-1].strip().split('-')[-1] in in_words:
                in_flag = True
            if w[-1].strip().split('-')[-1] != 'O' and w[-1].strip().split('-')[-1] in out_words:
                out_flag = True
        res.append(line) if in_flag and not out_flag else None
    return res

def get_multi_tagger(lines):
    res = []
    for line in lines:
        flag = False
        for w in line:
            if 'I-' in w[-1]:
                flag = True
        res.append(line) if flag else None
    return res

def write_format(lines, fn):
    with open(fn, 'w') as fout:
        for line in lines:
            for w in line:
                fout.write(w[0] + '\t' + w[-1] + '\n')
            fout.write('\n')

def write_labels(labels, fn):
    res = []
    for k in labels:
        res.append('B-' + k)
        res.append('I-' + k)
    res.append('O')
    with open(fn, 'wb') as fout:
        pickle.dump(res, fout)
        


if __name__ == '__main__':
    base = 'muc'
    few_words = few_words[base]
    
    with open(os.path.join(base, 'train.txt')) as fin:
        lines = fin.readlines()
    train = get_format(lines)
    print('train %d altogether' % len(train))
    with open(os.path.join(base, 'test.txt')) as fin:
        lines = fin.readlines()
    test = get_format(lines)
    print('test %d altogether' % len(test))
    all_words = get_labels(train)
    base_words = list(set(all_words) - set(few_words) - set(['O']))
    print(base_words)
    #write_labels(base_words, os.path.join(base, 'base_labels.pkl'))
    #write_labels(few_words, os.path.join(base, 'few_labels.pkl'))

    train_base_all = get_class(train, base_words, few_words)
    print('train base %d altogether' % len(train_base_all))
    #write_format(train_base_all, os.path.join(base, 'train_base.txt'))

    test_base = get_class(test, base_words, few_words)
    print('test base %d altogether' % len(test_base))
    #write_format(test_base, os.path.join(base, 'test_base.txt'))


    for ti in range(10):
        train_few_sample = []
        for k in few_words:
            print(k)
            train_few_single = get_class(train, [k], base_words)
            print(len(train_few_single))
            #train_few_multi = get_multi_tagger(train_few_single)
            #print(len(train_few_multi))
            ind = np.random.choice(len(train_few_single), 1)
            train_few_sample.extend([train_few_single[i] for i in ind])
            #ind = np.random.choice(len(train_few_multi), 1)
            #train_few_sample.extend([train_few_multi[i] for i in ind])
        random.shuffle(train_few_sample)
        print(len(train_few_sample))
        write_format(train_few_sample, os.path.join(base, 'train_few%d.txt'%(ti + 10)))

    test_few_all = get_class(test, few_words, base_words)
    print('test few all %d altogether' % len(test_few_all))
    #write_format(test_few_all, os.path.join(base, 'test_few.txt'))
        
    
    
