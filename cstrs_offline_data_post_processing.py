# [depends] %LIB%/controller_evaluation.py
"""
Post process the offline generated dataset.
"""
import sys
sys.path.append('../lib/')
from python_utils import H5pyTool
import os
import itertools
import numpy as np
from controller_evaluation import _post_process_data

_post_process_data(data_filename='cstrs_offline_data.h5py', 
                   num_data_gen_task=int(sys.argv[1]),
                   num_process_per_task=1)