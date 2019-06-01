from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt

import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency, metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score
import helper
np.random.seed(247)
tf.set_random_seed(247)

#---------------------------------------------------------------------------------------------------------



all_models = ['DistNet', 'LocalNet'] 
dropout_status = [True, False]
l2_status =      [True, False]
bn_status =      [True, False]
noise_status =   [False, True, False]
adv_status =     [False, False, True]

methods = ['backprop', 'smooth']


# save path
results_path = '../results'
params_path = utils.make_directory(results_path, 'model_params')


# dataset path
data_path = '../data/Synthetic_dataset.h5'
train, valid, test = helper.load_synthetic_dataset(data_path)

test_model = helper.load_synthetic_models(data_path, dataset='test')

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None

true_index = np.where(test['targets'][:,0] == 1)[0]
X = test['inputs'][true_index]
X_model = test_model[true_index]

for method in methods:
    print(method)
    backprop_results ={}
    for model_name in all_models:

        # model save path
        model_path = utils.make_directory(params_path, model_name)
        
        for n, noise in enumerate(noise_status):
            for i in range(len(dropout_status)):

                name = model_name
                if dropout_status[i]:
                    name += '_do'
                if l2_status[i]:
                    name += '_l2'
                if bn_status[i]:
                    name += '_bn'
                if noise:
                    name += '_noise'
                if adv_status[n]:
                    name += '_adv'
                print(name)
                
                file_path = os.path.join(model_path, name)

                # saliency parameters
                params = {'model_name': model_name, 
                          'input_shape': input_shape, 
                          'dropout_status': dropout_status[i],
                          'l2_status': l2_status[i],
                          'bn_status': bn_status[i],
                          'model_path': file_path+'_best.ckpt',
                         }

                if method == 'smooth':
                    X_saliency = helper.smooth_backprop(X, params, layer='output', class_index=None, num_average=50)
                else:
                    X_saliency = helper.backprop(X, params, layer='output', class_index=None, method=method)

                pr_score = []
                roc_score = []
                for j, gs in enumerate(X_saliency):
                    grad_times_input = np.squeeze(np.sum(X[j]*X_saliency[j], axis=2))

                    # calculate information of ground truth
                    gt_info = np.log2(4) - np.sum(-X_model[j]*np.log2(X_model[j]+1e-10),axis=0)

                    # set label if information is greater than 0
                    label = np.zeros(gt_info.shape)
                    label[gt_info > 0] = 1

                    # precision recall metric
                    precision, recall, thresholds = precision_recall_curve(label, grad_times_input)
                    pr_score.append(auc(recall, precision))

                    # roc curve
                    fpr, tpr, thresholds = roc_curve(label, grad_times_input)
                    roc_score.append(auc(fpr, tpr))

                backprop_results[name] = [np.array(roc_score), np.array(pr_score)]

    with open(os.path.join(results_path, method+'_interpretability_scores.pickle'), 'wb') as f:
        cPickle.dump(backprop_results, f, protocol=cPickle.HIGHEST_PROTOCOL)

