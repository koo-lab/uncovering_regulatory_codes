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

import copy
import helper
import time

np.random.seed(247)
tf.set_random_seed(247)

#------------------------------------------------------------------------------------------------

def perturb(x_nat, y, sess, nnmodel, feed_dict, grad_tensor, k=20):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    epsilon = 0.2
    x = np.copy(x_nat)
    feed_dict[xx] = x
    feed_dict[yy] = y
    feed_dict[is_train] = False
    
    for i in range(k):
        feed_dict[xx] = x
        grad = sess.run(grad_tensor, feed_dict=feed_dict)

        x += 0.1 /(i+10) * np.sign(grad)

        x = np.clip(x, x_nat - epsilon, x_nat + epsilon) 
        x = np.clip(x, 0, 1) # ensure valid pixel range

    feed_dict[is_train] = True
    return x

def initialize_feed_dict(placeholders, feed_dict):

    train_feed = {}
    for key in feed_dict.keys():
        train_feed[placeholders[key]] = feed_dict[key]
    return train_feed


#-------------------------------------------------------------------------------------------

model_name = 'DistNet'  #  'LocalNet' 
dropout_status = True
l2_status =      True
bn_status =      True
noise =          False


batch_size = 50
verbose = 2
num_epochs = 100
num_clean_epochs = 20


# save path
results_path = '../results'
params_path = utils.make_directory(results_path, 'model_params')

# dataset path
data_path = '../data/Synthetic_dataset.h5'
train, valid, test = helper.load_synthetic_dataset(data_path)

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None

adv_test = copy.deepcopy(test)

#----------------------------------------------------------------------------

tf.reset_default_graph()

# compile neural trainer
name = model_name
if dropout_status:
    name += '_do'
if l2_status:
    name += '_l2'
if bn_status:
    name += '_bn'
if noise:
    name += '_noise'
name += '_adv'
print('model: ' + name)

model_path = utils.make_directory(params_path, model_name)
file_path = os.path.join(model_path, name)

# load model parameters
model_layers, optimization, _ = helper.load_model(model_name, 
                                                  input_shape,
                                                  dropout_status, 
                                                  l2_status, 
                                                  bn_status)

# build neural network class
nnmodel = nn.NeuralNet()
nnmodel.build_layers(model_layers, optimization, supervised=True)

grad_tensor = tf.gradients(nnmodel.mean_loss, nnmodel.placeholders['inputs'])[0]

xx = nnmodel.placeholders['inputs']
yy = nnmodel.placeholders['targets']
is_train = nnmodel.placeholders['is_training']
loss = nnmodel.mean_loss
#   nnmodel.inspect_layers()
performance = nn.MonitorPerformance('train', optimization['objective'], verbose)
performance.set_start_time(start_time = time.time())

train_calc = [nnmodel.train_step, loss, nnmodel.metric]
train_feed = initialize_feed_dict(nnmodel.placeholders, nnmodel.feed_dict)

# create neural trainer
nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

# initialize session
sess = utils.initialize_session()

# set data in dictionary
#   data = {'train': train, 'valid': valid, 'test': test}
x_train = train['inputs']
y_train = train['targets']


for epoch in range(num_epochs):
    print(epoch)
    index = np.random.permutation(len(x_train))

    for i in range((len(x_train) // batch_size)):
        # batch
        clean_batch_x = x_train[index[i*batch_size:(i+1)*batch_size]]
        clean_batch_y = y_train[index[i*batch_size:(i+1)*batch_size]]


        if epoch >= num_clean_epochs:
            adv_batch = perturb(clean_batch_x, clean_batch_y,
                        sess, nnmodel, train_feed, grad_tensor)


            train_feed[xx] = np.concatenate([clean_batch_x, adv_batch])
            train_feed[yy] = np.concatenate([clean_batch_y, clean_batch_y])
        else:
            train_feed[xx] = clean_batch_x
            train_feed[yy] = clean_batch_y

        results = sess.run(train_calc, feed_dict=train_feed)
        performance.add_loss(results[1])
        #performance.progress_bar(i+1., (len(x_train) // batch_size), metric/(i+1))

    predictions = nntrainer.get_activations(sess, valid, 'output')
    print(metrics.accuracy(valid['targets'], predictions))

    if epoch >= num_clean_epochs:
        adv_test['inputs'] = perturb(test['inputs'], test['targets'], sess, nnmodel, train_feed, grad_tensor)
        predictions = nntrainer.get_activations(sess, adv_test, 'output')
        roc, roc_curves = metrics.roc(test['targets'], predictions)
        print('Adversarial Accuracy')
        print(metrics.accuracy(test['targets'], predictions))


    # save cross-validcation metrics
    loss, mean_vals, error_vals = nntrainer.test_model(sess, valid,
                                                            name="valid",
                                                            batch_size=batch_size,
                                                            verbose=verbose)

# save cross-validcation metrics
loss, mean_vals, error_vals = nntrainer.test_model(sess, test,
                                                        name="test",
                                                        batch_size=batch_size,
                                                        verbose=verbose)

#nntrainer.save_model(sess)
nnmodel.save_model_parameters(sess, file_path+'_best.ckpt')





