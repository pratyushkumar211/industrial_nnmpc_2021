# [depends] cstrs_parameters.pickle cstrs_train.pickle
# [depends] %LIB%/controller_evaluation.py
# [makes] pickle
"""
Script to run an on-line simulation for the double integrator example
using the approximate NN controller.
"""
import sys
sys.path.append('lib/')
from controller_evaluation import _simulate_neural_networks

# Load data and do an online simulation using the linear MPC controller.
_simulate_neural_networks(plant_name='cstrs', Nsim=4320)