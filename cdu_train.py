# [depends] %LIB%/LinearMPCLayers.py
# [depends] %LIB%/controller_evaluation.py
# [depends] %LIB%/python_utils.py
# [makes] pickle
"""
Script to train the neural network controller for the 
CDU example.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import os
import sys
import numpy as np
sys.path.append('lib/')
import time
import itertools
import tensorflow as tf
from python_utils import (PickleTool, H5pyTool)
from LinearMPCLayers import RegulatorModel
from controller_evaluation import _get_data_for_training

# Set the tensorflow graph-level seed.
tf.random.set_seed(1)

def create_nn_controller(*, regulator_dims):
    """ Create an instantiation of 
        the neural network controller and return."""
    # Get the shapes.
    Nx = 252
    Nu = 32
    # Create the keras model instance.
    nn_controller = RegulatorModel(Nx=Nx, Nu=Nu,
                                   regulator_dims=regulator_dims,
                                   nnwithuprev=False)
    # Compile the nn controller.
    nn_controller.compile(optimizer='adam', loss='mean_squared_error')
    # Return the compiled controller.
    return nn_controller

def train_nn_controller(nn_controller, data, 
                        stdout_filename, ckpt_path):
    """ Function to train the NN controller."""
    # Std out.
    sys.stdout = open(stdout_filename, 'a')
    # Create the checkpoint callback.
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    verbose=1)
    
    # Call the fit method to train.
    tstart = time.time()
    nn_controller.fit(x=[data['x'], data['xs'], data['us']], 
                      y=[data['u']], 
                      epochs=1500,
                      batch_size=2048,
                      validation_split=0.05,
                      callbacks = [checkpoint_callback])
    tend = time.time()
    training_time = tend - tstart
    # Return the NN controller.
    return (nn_controller, training_time)

# Get the raw data for training.
raw_data = H5pyTool.load_training_data(filename='cdu_offline_data.h5py')
(scaled_data, xscale) = _get_data_for_training(data=raw_data, 
                                           num_samples=357600, 
                                           scale=True)

# Arrange the number of samples and the dimensions of the NN controller.
num_samples = [20000]
num_samples += [num_sample for num_sample in range(30000, 330001, 30000)]
num_samples += [357600]
data_generation_times = []
training_times = []
memory_footprints = []
regulator_dims = [[536, 832, 832, 832, 32],
                  [536, 896, 896, 896, 32],
                  [536, 960, 960, 960, 32],
                  [536, 1024, 1024, 1024, 32]]
trained_regulator_weights = []

# Take input from the user about which architecture to train.
train_architecture = int(sys.argv[1]) - 1
regulator_dims = [regulator_dims[train_architecture]]

# Loop over all the desired training scenarios.
for (regulator_dim, num_sample) in itertools.product(regulator_dims, 
                                                     num_samples):
    # Get the training data.
    training_data = _get_data_for_training(data=scaled_data, 
                                            num_samples=num_sample,
                                            scale=False)
    # Get a NN controller.
    nn_controller = create_nn_controller(regulator_dims=regulator_dim)
    # Train the NN controller.
    (nn_controller, training_time) = train_nn_controller(nn_controller, 
                                        training_data,
                    stdout_filename=str(train_architecture)+'-cdu_train.txt',
                    ckpt_path=str(train_architecture)+'-cdu_train.ckpt')
    # Load and save the best weights.
    nn_controller.load_weights(str(train_architecture)+'-cdu_train.ckpt')
    trained_regulator_weights.append(nn_controller.get_weights())
    # Get the overall NN design time.
    data_generation_time = num_sample/num_samples[-1]
    data_generation_time = data_generation_time*raw_data['data_gen_time']
    data_generation_times.append(data_generation_time)
    training_times.append(training_time)
    # Get the NN memory footprints.
    memory_footprint = PickleTool.get_nn_memory_footprint(regulator_weights=
                            trained_regulator_weights[-1], 
            filename=str(train_architecture)+'-cdu_trained_weights.pickle')
    memory_footprints.append(memory_footprint)

# Save the weights.
cdu_training_data = dict(trained_regulator_weights=trained_regulator_weights,
                         num_architectures=len(regulator_dims),
                         num_samples=num_samples,
                         data_generation_times=data_generation_times, 
                         training_times=training_times,
                         memory_footprints=memory_footprints,
                         xscale=xscale,
                         regulator_dims=regulator_dims)
# Save data.
PickleTool.save(data_object=cdu_training_data, 
                filename=str(train_architecture)+'-cdu_train.pickle')
