""" Custom python utilities. 
Pratyush Kumar, pratyushkumar@ucsb.edu
"""

import numpy as np
import pickle
import h5py 
import os

figure_size_a4 = (8, 10)
figure_size_cl = (8, 8)
figure_size_metrics = (8, 5)

class PickleTool:
    """Class which contains a few static methods for saving and
    loading pkl data files conveniently."""
    @staticmethod
    def load(filename, type='write'):
        """Wrapper to load data."""
        if type == 'read':
            with open(filename, "rb") as stream:
                return pickle.load(stream)
        if type == 'write':
            with open(filename, "wb") as stream:
                return pickle.load(stream)
    
    @staticmethod
    def save(data_object, filename):
        """Wrapper to pickle a data object."""
        with open(filename, "wb") as stream:
            pickle.dump(data_object, stream)

    @staticmethod
    def get_nn_memory_footprint(regulator_weights, filename):
        """Save the NN weights and get the memomry footprint
           in kB."""
        with open(filename, "wb") as stream:
            pickle.dump(regulator_weights, stream)
        return os.path.getsize(filename)/1024

class H5pyTool:
    """Class which contains a few static methods for saving and
    loading h5py data files conveniently."""
    @staticmethod
    def load_training_data(filename):
        """Wrapper to load data."""
        with h5py.File(filename, "r") as stream:
            dictionary = {}
            for key in list(stream.keys()):
                dictionary[key] = np.asarray(stream.get(key))
            return dictionary
    
    @staticmethod
    def save_training_data(dictionary, filename):
        """Wrapper to pickle a data object."""
        with h5py.File(filename, "w") as stream:
            for (key, value) in dictionary.items():
                stream.create_dataset(key, data=value)