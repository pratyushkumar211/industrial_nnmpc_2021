# [depends] cdu_parameters.pickle
# [depends] %LIB%/python_utils.py
# [makes] pickle
"""
Script to generate offline data for training the neural net.
"""
import sys
sys.path.append('../lib/')
from python_utils import PickleTool

# Load data and do an online simulation using the linear MPC controller.
task_number = int(sys.argv[1]) - 1
cdu_parameters = PickleTool.load(filename='cdu_parameters.pickle', 
                                 type='read')
cdu_parameters['offline_simulator'].generate_data(task_number=task_number,   
                    data_filename='cdu_offline_data.h5py',
                    stdout_filename=str(task_number)+'-cdu_offline_data.txt')