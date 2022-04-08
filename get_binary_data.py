import pickle
import os
import random
import numpy as np

pos_indices = list(range(1, 27))
other_indice = 27
base_dir = 'data/ontonotes/'
res = {}
other = []
with open(os.path.join(base_dir, 'imprint_few.pkl'), 'rb') as fin:
    p1, p2, p3, p4 = pickle.load(fin)

for i, (features, labels) in enumerate(list(zip(p1, p2))):
    for j, (feature, label) in enumerate(list(zip(features, labels))):
        if label in pos_indices:
            key = (label - 1) // 2
            cr = res.get(key, [])
            cr.append(feature)
            res[key] = cr
        if label == other_indice:
            other.append([feature, (i, j)])

for k in res:
    print(len(res[k]))

#centers = [np.mean(res[k], axis = 0) for k in res]
#with open(os.path.join(base_dir, 'centers.pkl'), 'wb') as fout:
#    pickle.dump(centers, fout)
print('------------')
print(len(other))


datasets = []
for k in res:
    inds1 = np.random.choice(len(res[k]), size = 400)
    inds2 = np.random.choice(len(res[k]), size = 400)
    for i in inds1:
        for j in inds2:
            datasets.append([np.concatenate([res[k][i], res[k][j], np.abs(res[k][i] - res[k][j]),
                np.multiply(res[k][i], res[k][j])], axis = -1), 1])

print(len(datasets))
keys = list(res.keys())

for _ in range(1500):
    key1 = random.choice(keys)
    key2 = random.choice(keys)
    while key2 == key1:
        key2 = random.choice(keys)
    inds1 = np.random.choice(len(res[key1]), size = 50)
    inds2 = np.random.choice(len(res[key2]), size = 50)
    for i in inds1:
        for j in inds2:
            datasets.append([np.concatenate([res[key1][i], res[key2][j], np.abs(res[key1][i] - res[key2][j]),
                np.multiply(res[key1][i], res[key2][j])], axis = -1), 0])

with open(os.path.join(base_dir, 'binary_train.pkl'), 'wb') as fout:
    pickle.dump(datasets, fout)

with open(os.path.join(base_dir, 'binary_test.pkl'), 'wb') as fout:
    pickle.dump(other, fout)
