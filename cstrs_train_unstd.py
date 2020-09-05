# [makes] pickle
"""
Script to train an unstructured neural network controller for the 
double integrator example.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
sys.path.append('lib/')
import numpy as np
import os
import time
import itertools
import tensorflow as tf
from python_utils import (PickleTool, H5pyTool)
from LinearMPCLayers import UnstdRegulatorModel
from controller_evaluation import _get_data_for_training

# Set the tensorflow graph-level seed.
tf.random.set_seed(1)

def create_nn_controller(*, regulator_dims):
    """Create an instantiation of the neural network controller and return."""
    Nx = 12
    Nu = 6
    # Create the keras model instance.
    nn_controller = UnstdRegulatorModel(Nx=Nx, Nu=Nu,
                                       regulator_dims=regulator_dims,
                                       nnwithuprev=True)
    # Compile the nn controller.
    nn_controller.compile(optimizer='adam', loss='mean_squared_error')
    # Return the compiled controller.
    return nn_controller

def train_nn_controller(nn_controller, data, stdout_filename, ckpt_path):
    """ Function to train the controller."""
    # Std out.
    sys.stdout = open(stdout_filename, 'w')
    # Create the checkpoint callback.
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                monitor='val_loss',
                                                save_best_only=True,
                                                save_weights_only=True,
                                                verbose=1)
    # Call the fit method to train.
    tstart = time.time()
    nn_controller.fit(x=[data['x'], data['uprev'], data['xs'], data['us']], 
                      y=[data['u']], 
                      epochs=2000,
                      batch_size=1024,
                      validation_split=0.1,
                      callbacks=[checkpoint_callback])
    tend = time.time()
    training_time = tend - tstart
    return (nn_controller, training_time)

# Get the raw data for training.
num_sample = 150000
raw_data = H5pyTool.load_training_data(filename='cstrs_offline_data.h5py')
(training_data, xscale) = _get_data_for_training(data=raw_data, 
                                           num_samples=num_sample, 
                                           scale=True)

# Get a NN controller.
regulator_dims = [36, 224, 224, 224, 6]
nn_controller = create_nn_controller(regulator_dims=regulator_dims)

# Train the NN controller.
(nn_controller, training_time) = train_nn_controller(nn_controller, 
                                                     training_data,
                              stdout_filename='cstrs_train_unstd.txt',
                              ckpt_path='cstrs_train_unstd.ckpt')

# Get the best weights.
nn_controller.load_weights('cstrs_train_unstd.ckpt')
trained_regulator_weights = nn_controller.get_weights()

# Get the NN memory footprints.
memory_footprint = PickleTool.get_nn_memory_footprint(regulator_weights=
                            trained_regulator_weights, 
                filename='cstrs_trained_weights_unstd.pickle')

# Save the weights.
cstrs_training_data = dict(trained_regulator_weights=trained_regulator_weights,
                        training_time=training_time,
                        memory_footprint=memory_footprint,
                        xscale=xscale,
                        regulator_dims=regulator_dims)
# Save data.
PickleTool.save(data_object=cstrs_training_data, 
                filename='cstrs_train_unstd.pickle')