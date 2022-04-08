import pickle
import os
import numpy as np

from scipy.special import softmax
from sklearn.cluster import KMeans

base_dir = 'data/ontonotes'

with open(os.path.join(base_dir, 'other_sims.pkl'), 'rb') as fin:
    sims = pickle.load(fin)

features = []
labels = []

for sim in sims:
    feature = softmax(sim[0], axis = -1)[:, 1]
    features.append(feature)
    labels.append(sim[1])

stats = {}
res = {}
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(features)
for l, p in zip(kmeans.labels_, labels):
    cr = res.get(l, [])
    cr.append(p)
    res[l] = cr
    stats[l] = stats.get(l, 0) + 1
print(stats)
for k in res:
    res[k] = np.mean(res[k], axis = 0)

other_protos = np.asarray([res[k] for k in range(5)])
with open(os.path.join(base_dir, 'other_protos.pkl'), 'wb') as fout:
    pickle.dump(other_protos, fout)


