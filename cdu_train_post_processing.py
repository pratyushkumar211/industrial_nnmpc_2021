# [depends] %LIB%/controller_evaluation.py
"""
Post process the trained dataset.
"""
import sys
sys.path.append('lib/')
from controller_evaluation import _post_process_trained_data

_post_process_trained_data(data_filename='cdu_train.pickle', 
                           num_architectures=int(sys.argv[1]))
