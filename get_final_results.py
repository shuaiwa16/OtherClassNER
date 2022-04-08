import pickle
import numpy as np
import os
import tensorflow as tf

from sklearn.metrics import f1_score,precision_score,recall_score

base_dir = 'data/ontonotes'

ckpt_name = 'result_dir_trans/model.ckpt-2334'
other_indice = 27
few_num = 10

with open(os.path.join(base_dir, 'results.pickle'), 'rb') as fin:
    p1, p2, p3, p4 = pickle.load(fin)

num = 0
acc = 0

all_predictions = []
all_labels = []

current_weights = tf.train.load_variable(ckpt_name, 'output_weights')
for features, labels in zip(p1, p2):
    logits = np.matmul(features, np.transpose(current_weights[other_indice + 1:, :]))
    preds = np.argmax(logits, axis = -1)
    for pred, label in zip(preds, labels):
        #if label == 27:
        #    print(pred)
        if label >= other_indice:
            num += 1
            if pred > few_num + 2:
                pred = -1
            all_predictions.append(pred + 1)
            all_labels.append(label - other_indice)
            if label - other_indice - 1 == pred:
                acc += 1

print(set(all_predictions) - set(all_labels))
print('accuracy:', acc / float(num))
print('precision:', precision_score(all_labels, all_predictions, average = 'macro'))
print('recall:', recall_score(all_labels, all_predictions, average = 'macro'))
print('f score:', f1_score(all_labels, all_predictions, average = 'macro'))
    
