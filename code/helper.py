from __future__ import print_function

import os, sys
import h5py
import numpy as np

import tensorflow as tf

sys.path.append('../..')
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency, metrics


def load_synthetic_dataset(filepath, verbose=True):
	# setup paths for file handling

	trainmat = h5py.File(filepath, 'r')

	if verbose:
		print("loading training data")
	X_train = np.array(trainmat['X_train']).astype(np.float32)
	y_train = np.array(trainmat['Y_train']).astype(np.float32)

	if verbose:
		print("loading cross-validation data")
	X_valid = np.array(trainmat['X_valid']).astype(np.float32)
	y_valid = np.array(trainmat['Y_valid']).astype(np.int32)

	if verbose:
		print("loading test data")
	X_test = np.array(trainmat['X_test']).astype(np.float32)
	y_test = np.array(trainmat['Y_test']).astype(np.int32)


	X_train = np.expand_dims(X_train, axis=3).transpose([0,2,3,1])
	X_valid = np.expand_dims(X_valid, axis=3).transpose([0,2,3,1])
	X_test = np.expand_dims(X_test, axis=3).transpose([0,2,3,1])

	train = {'inputs': X_train, 'targets': y_train}
	valid = {'inputs': X_valid, 'targets': y_valid}
	test = {'inputs': X_test, 'targets': y_test}

	return train, valid, test


def load_synthetic_models(filepath, dataset='test'):
	# setup paths for file handling

	trainmat = h5py.File(filepath, 'r')
	if dataset == 'train':
		return np.array(trainmat['model_train']).astype(np.float32)
	elif dataset == 'valid':
		return np.array(trainmat['model_valid']).astype(np.float32)
	elif dataset == 'test':
		return np.array(trainmat['model_test']).astype(np.float32)



def load_model(model_name, input_shape, 
				dropout_status=True, l2_status=True, bn_status=True):

	# import model
	if model_name == 'DistNet':
		from models import DistNet as genome_model
	elif model_name == 'StandardNet':
		from models import StandardNet as genome_model
	elif model_name == 'LocalNet':
		from models import LocalNet as genome_model
	elif model_name == 'DeepBind':
		from models import DeepBind as genome_model

	# load model specs
	model_layers, optimization = genome_model.model(input_shape,
													dropout_status,
													l2_status,
													bn_status)

	return model_layers, optimization, genome_model



def backprop(X, params, layer='output', class_index=None, batch_size=128, method='guided'):
	"""wrapper for backprop/guided-backpro saliency"""

	tf.reset_default_graph()

	# build new graph
	model_layers, optimization, genome_model = load_model(params['model_name'], params['input_shape'], 
												   params['dropout_status'], params['l2_status'], params['bn_status'])

	nnmodel = nn.NeuralNet()
	nnmodel.build_layers(model_layers, optimization, method=method, use_scope=True)
	nntrainer = nn.NeuralTrainer(nnmodel, save='best', filepath=params['model_path'])

	# setup session and restore optimal parameters
	sess = utils.initialize_session(nnmodel.placeholders)
	nntrainer.set_best_parameters(sess, params['model_path'], verbose=0)

	# backprop saliency
	if layer == 'output':
		layer = list(nnmodel.network.keys())[-2]

	saliency = nntrainer.get_saliency(sess, X, nnmodel.network[layer], class_index=class_index, batch_size=batch_size)

	sess.close()
	tf.reset_default_graph()
	return saliency



def smooth_backprop(X, params, layer='output', class_index=None, batch_size=128, num_average=100):
    """wrapper for backprop/guided-backpro saliency"""


    saliency = np.zeros(X.shape)
    for i, x in enumerate(X):
        if np.mod(i,200) == 0:
            print('%d out of %d'%(i, len(X)))

        if np.mod(i,50) == 0:
            tf.reset_default_graph()

            # build new graph
            model_layers, optimization, genome_model = load_model(params['model_name'], params['input_shape'], 
                                                           params['dropout_status'], params['l2_status'], params['bn_status'])

            nnmodel = nn.NeuralNet()
            nnmodel.build_layers(model_layers, optimization, method='backprop', use_scope=True)
            nntrainer = nn.NeuralTrainer(nnmodel, save='best', filepath=params['model_path'])

            # setup session and restore optimal parameters
            sess = utils.initialize_session(nnmodel.placeholders)
            nntrainer.set_best_parameters(sess, params['model_path'], verbose=0)

            # backprop saliency
            if layer == 'output':
                layer = list(nnmodel.network.keys())[-2]

        x = np.expand_dims(x, axis=0)
        shape = list(x.shape)
        shape[0] = num_average
        
        noisy_saliency = nntrainer.get_saliency(sess, x+np.random.normal(scale=0.1, size=shape), nnmodel.network[layer], class_index=class_index, batch_size=num_average)
        saliency[i,:,:,:] = np.mean(noisy_saliency, axis=0)

        #if np.mod(i,99) == 0:
        #   sess.close()
        #   tf.reset_default_graph()

    return saliency


def entropy_weighted_distance(X_saliency, X_model):
	"""calculate entropy-weighted distance between normalized saliency map and model"""
	def cosine_distance(X_norm, X_model):
		norm1 = np.sqrt(np.sum(X_norm**2, axis=0))
		norm2 = np.sqrt(np.sum(X_model**2, axis=0))

		dist = np.sum(X_norm*X_model, axis=0)/norm1/norm2
		return dist

	def entropy(X):
		information = np.log2(4) - np.sum(-X*np.log2(X+1e-10),axis=0)
		return information

	X_norm = utils.normalize_pwm(X_saliency, factor=3)
	cd = cosine_distance(X_norm, X_model)
	model_info = entropy(X_model)
	tpr = np.sum(model_info*cd)/np.sum(model_info)

	inv_model_info = -(model_info-2)
	inv_cd = -(cd-1)
	fpr = np.sum(inv_cd*inv_model_info)/np.sum(inv_model_info)

	return tpr, fpr



