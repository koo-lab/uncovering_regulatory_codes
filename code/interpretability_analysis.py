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

import helper

np.random.seed(247)
tf.set_random_seed(247)

#-------------------------------------------------------------------------------------

# attribution methods
methods = ['backprop', 'smoothgrad']# , 'guided']

all_models = ['DistNet', 'LocalNet'] # 'StandardNet', 'DeepBind', 

# regularization settings
dropout_status = [True, False]
l2_status =      [True, False]
bn_status =      [True, False]

# add gaussian noise
add_noise =      [True, False]

# save path
results_path = '../results'
params_path = utils.make_directory(results_path, 'model_params')

#-------------------------------------------------------------------------------------

# load dataset & ground truth
data_path = '../data/Synthetic_dataset.h5'
train, valid, test = helper.load_synthetic_dataset(data_path)
test_model = helper.load_synthetic_models(data_path, dataset='test')
    
# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = [None, train['targets'].shape[1]]

# perform analysis only on positive label sequences
true_index = np.where(test['targets'][:,0] == 1)[0]
X = test['inputs'][true_index]
X_model = test_model[true_index]

#-------------------------------------------------------------------------------------

for method in methods:

    backprop_results ={}
    for model_name in all_models:
        
        for noise in add_noise:

            for i in range(len(dropout_status)):

                # set name
                name = model_name
                if dropout_status[i]:
                    name += '_do'
                if l2_status[i]:
                    name += '_l2'
                if bn_status[i]:
                    name += '_bn'
                if noise:
                    name += '_noise'
                
                file_path = os.path.join(params_path, name)

                # attribution parameters
                params = {'model_name': model_name, 
                          'input_shape': input_shape, 
                          'dropout_status': dropout_status[i],
                          'l2_status': l2_status[i],
                          'bn_status': bn_status[i],
                          'model_path': file_path+'_best.ckpt',
                         }

                # calculate attribution scores
                if method == 'smoothgrad':
                    attribution_score = helper.smooth_backprop(X, params, layer='output', class_index=None, num_average=50)
                else:
                    attribution_score = helper.backprop(X, params, layer='output', class_index=None, method=method)

                # calculate TPS and FPS 
                info = []
                for j, gs in enumerate(attribution_score):
                    X_saliency = np.squeeze(gs).T
                    tps, fps = helper.entropy_weighted_distance(X_saliency, X_model[j])
                    info.append([tps, fps])
                info = np.array(info)

                backprop_results[name] = info

    # save results
    with open(os.path.join(results_path, method+'.pickle'), 'wb') as f:
        cPickle.dump(backprop_results, f, protocol=cPickle.HIGHEST_PROTOCOL)