import tensorflow as tf
import pickle
import os
import numpy as np

ckpt_name = 'result_dir/model.ckpt-2334'
few_thres = 31
base_dir = 'data/ontonotes'
num_few = 10

current_weights = tf.train.load_variable(ckpt_name, 'output_weights')
print(current_weights.shape)

with open(os.path.join(base_dir, 'imprint_few.pkl'), 'rb') as fin:
    p1, p2, p3, p4 = pickle.load(fin)

stats = {}
for i, (features, labels) in enumerate(list(zip(p1, p2))):
    for feature, label in zip(features, labels):
        if label >= few_thres:
            cr = stats.get(label, [])
            cr.append(feature)
            stats[label] = cr
for k in stats:
    print(k)
    print(len(stats[k]))
    stats[k] = np.mean(stats[k], axis = 0)


few_weights = np.asarray([stats.get(i, np.random.rand(768)) for i in range(few_thres, few_thres + num_few)])
current_weights = np.concatenate([current_weights[:few_thres, :], few_weights], axis = 0)
print(current_weights.shape)

with open(os.path.join(base_dir, 'other_protos.pkl'), 'rb') as fin:
    other_protos = pickle.load(fin)
print(other_protos.shape)
few_weights = np.concatenate([current_weights, other_protos], axis = 0)
print(few_weights.shape)
with open(os.path.join(base_dir, 'proto.pkl'), 'wb') as fout:
    pickle.dump(current_weights, fout)


