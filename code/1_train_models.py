from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import helper
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit

#------------------------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------------------------

# load dataset
data_path = '../data/Synthetic_dataset.h5'
train, valid, test = helper.load_synthetic_dataset(data_path)

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None

#-------------------------------------------------------------------------------------
# train models

for i in range(len(dropout_status)):
    for noise in noise_status:
        for model_name in all_models:
            tf.reset_default_graph()
            print('model: ' + model_name)

            # compile neural trainer
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

            # load model parameters
            model_layers, optimization, _ = helper.load_model(model_name, 
                                                              input_shape,
                                                              dropout_status[i], 
                                                              l2_status[i], 
                                                              bn_status[i])

            # build neural network class
            nnmodel = nn.NeuralNet(seed=247)
            nnmodel.build_layers(model_layers, optimization, supervised=True)

            nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

            # initialize session
            sess = utils.initialize_session()

            # set data in dictionary
            data = {'train': train, 'valid': valid, 'test': test}
        
            # set data in dictionary
            num_epochs = 200
            batch_size = 100
            patience = 20
            verbose = 2
            shuffle = True
            for epoch in range(num_epochs):
                if verbose >= 1:
                    sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))
                else:
                    if epoch % 10 == 0:
                        sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1,num_epochs))

                # add noise to inputs
                if noise:       
                    noisy_train = {'inputs': train['inputs'] + np.random.normal(scale=0.1, size=train['inputs'].shape),
                                   'targets': train['targets']}
                else:
                    noisy_train = train

                # train epoch
                train_loss = nntrainer.train_epoch(sess, noisy_train,
                                                    batch_size=batch_size,
                                                    verbose=verbose,
                                                    shuffle=shuffle)

                # save cross-validcation metrics
                loss, mean_vals, error_vals = nntrainer.test_model(sess, valid,
                                                                        name="valid",
                                                                        batch_size=batch_size,
                                                                        verbose=verbose)
                # save model
                nntrainer.save_model(sess)

                # early stopping
                if not nntrainer.early_stopping(loss, patience):
                    break

            nntrainer.save_model(sess, addon='last')
