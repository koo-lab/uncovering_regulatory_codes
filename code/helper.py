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
    elif model_name == 'LocalNet':
        from models import LocalNet as genome_model
    elif model_name == 'DistNet_exp':
        from models import DistNet_exp as genome_model
    elif model_name == 'LocalNet_exp':
        from models import LocalNet_exp as genome_model
    elif model_name == 'StandardNet':
        from models import StandardNet as genome_model
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

def entropy_weighted_cosine_distance(X_saliency, X_model):
    """calculate entropy-weighted cosine distance between normalized saliency map and model"""
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




#-------------------------------------------------------------------------------------
#First Order Mutagenesis Functions
def fom_heatmap(X, layer, alphabet, nntrainer, sess, eps=1e-7):
    '''Function that performs First Order Mutagenesis and returns an array of the saliency 
    output (thought of as a (4,seqlen) heatmap'''
    
    def mutate(sequence, seq_length, dims):
        num_mutations = seq_length * dims
        hotplot_mutations = np.zeros((num_mutations,seq_length,1,dims)) 

        for position in range(seq_length):
            for nuc in range(dims):
                mut_seq = np.copy(sequence)          
                mut_seq[0, position, 0, :] = np.zeros(dims)
                mut_seq[0, position, 0, nuc] = 1.0

                hotplot_mutations[(position*dims)+nuc] = mut_seq
        return hotplot_mutations

    #first mutate the sequence
    X_mut = mutate(X, X.shape[1], X.shape[3])

    #take all the mutations and assign them into a dict for deepomics
    mutations = {'inputs': X_mut, 'targets': np.ones((X_mut.shape[0], 1))}
    #Get output or logits activations for the mutations
    mut_predictions = nntrainer.get_activations(sess, mutations, layer=layer)

    #take the WT and put it into a dict for deepomics
    WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
    #Get output or logits activations for the WT sequence
    predictions = nntrainer.get_activations(sess, WT, layer=layer)

    #shape the predictions of the mutations into the shape of a heatmap
    heat_mut = mut_predictions.reshape(X.shape[1],4).T
    
    #normalize the heat map rearrangement by minusing it by the true prediction score of that test sequence
    #norm_heat_mut = heat_mut - predictions[0] + eps
    #norm_heat_mut = utils.normalize_pwm(norm_heat_mut, factor=4)

    return (heat_mut)

def analysis_names():
    '''The outputs of the fom_full analysis file is a single array with shape 
    (num_models, num_regmethods, num_poslabels, dims, seqlen). To simplify indexing into the array to select a single models results, this function outputs a pandas dataframe and a dictionary that allows one to qualitatively identify how they should index into the array.'''
    
    all_models = ['DistNet', 'LocalNet', 'DeepBind', 'StandardNet']
    num_models = len(all_models)
    dropout_status = [True, True,   False,  False,  False, True,  True,  False]
    l2_status =      [True, False,  True,   False,  False, True,  False, True]
    bn_status =      [True, False,  False,  True,   False, False, True,  True]
    num_reg = len(dropout_status)

    Names = []
    # loop through models
    for t, model_name in enumerate(all_models): 
        #Names[t] = []
        #loop through every regularization type
        modelsreg = []
        for i in range(len(dropout_status)):

            # compile neural trainer
            name = model_name
            if dropout_status[i]:
                name += '_do'
            if l2_status[i]:
                name += '_l2'
            if bn_status[i]:
                name += '_bn'
            modelsreg.append(name)
            #Names[t].append(name)
        Names.append(modelsreg)
        
    analysis_idx = {}
    for m, model in enumerate(Names):
        for r, reg in enumerate(model):
            analysis_idx[reg] = [m,r]

    import pandas as pd
    return (pd.DataFrame(Names), analysis_idx)
    


def clip_filters(W, threshold=0.5, pad=3):
    num_filters, _, filter_length = W.shape

    W_clipped = []
    for i in range(num_filters):
        w = utils.normalize_pwm(W[i], factor=3)
        entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=0)
        index = np.where(entropy > threshold)[0]
        if index.any():
            start = np.maximum(np.min(index)-pad, 0)
            end = np.minimum(np.max(index)+pad+1, filter_length)
            W_clipped.append(W[i,:,start:end])
        else:
            W_clipped.append(W[i,:,:])

    return W_clipped



def meme_generate(W, output_file='meme.txt', prefix='filter', factor=None):

    # background frequency
    nt_freqs = [1./4 for i in range(4)]

    # open file for writing
    f = open(output_file, 'w')

    # print intro material
    f.write('MEME version 4\n')
    f.write('\n')
    f.write('ALPHABET= ACGT\n')
    f.write('\n')
    f.write('Background letter frequencies:\n')
    f.write('A %.4f C %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
    f.write('\n')

    for j in range(len(W)):
        if factor:
            pwm = utils.normalize_pwm(W[j], factor=factor)
        else:
            pwm = W[j]
        f.write('MOTIF %s%d \n' % (prefix, j))
        f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (pwm.shape[1], pwm.shape[1]))
        for i in range(pwm.shape[1]):
            f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[:,i]))
        f.write('\n')

    f.close()

