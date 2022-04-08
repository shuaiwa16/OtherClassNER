import tensorflow as tf
import pickle
import os
import numpy as np

from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

base_dir = 'data/ontonotes'

with open(os.path.join(base_dir, 'binary_train.pkl'), 'rb') as fin:
    binary_train = pickle.load(fin)

print('load data done')

batch_size = 4096
features = tf.placeholder(dtype = tf.float32, shape = [None, 3072])
labels = tf.placeholder(dtype = tf.int32, shape = [None])
onehot_labels = tf.one_hot(labels, 2)
h1 = tf.layers.dense(features, 4096)
h1 = tf.nn.dropout(h1, 0.5)
h1 = tf.nn.relu(h1)
h2 = tf.layers.dense(h1, 512)
h2 = tf.nn.dropout(h2, 0.5)
h2 = tf.nn.relu(h2)
h3 = tf.layers.dense(h2, 32)
h3 = tf.nn.dropout(h3, 0.5)
h3 = tf.nn.relu(h3)
h4 = tf.layers.dense(h3, 2)

def get_features(x, y):
    return np.concatenate([x, y, np.abs(x - y), np.multiply(x, y)], axis = -1)

preds = tf.cast(tf.argmax(h4, axis = -1), tf.int32)
loss = tf.nn.softmax_cross_entropy_with_logits(logits = h4, labels = onehot_labels)
loss = tf.reduce_mean(loss)
acc = tf.reduce_sum(tf.cast(tf.equal(labels, preds), tf.float32)) / batch_size
opt = tf.train.AdamOptimizer().minimize(loss)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    inds = np.random.choice(len(binary_train), size = batch_size)
    batch_features = [binary_train[t][0] for t in inds]
    batch_labels = [binary_train[t][1] for t in inds]
    feed_dict = {features:batch_features, labels:batch_labels}
    for i in range(10001):
        lv, av, _ = sess.run([loss, acc, opt], feed_dict = feed_dict)
        if i % 1000 == 0:
            print('iter %d, loss:%f, accuracy:%f'%(i, lv, av))
        if i % 10000 == 0:
            saver.save(sess, os.path.join(base_dir, 'binary_model_%d.ckpt'%i))
    with open(os.path.join(base_dir, 'binary_test.pkl'), 'rb') as fin:
        others = pickle.load(fin)
    with open(os.path.join(base_dir, 'centers.pkl'), 'rb') as fin:
        centers = pickle.load(fin)
    sims = []
    for other in tqdm(others):
        current_features = [get_features(other[0], center) for center in centers]
        logits = sess.run(h4, feed_dict = {features:current_features})
        sims.append([logits, other[0]])
    with open(os.path.join(base_dir, 'other_sims.pkl'), 'wb') as fout:
        pickle.dump(sims, fout)


