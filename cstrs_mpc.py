# [depends] cstrs_parameters.pickle
# [depends] %LIB%/controller_evaluation.py
# [depends] %LIB%/python_utils.py
# [makes] pickle
"""
Script to run an on-line simulation using the double integrator example
using the optimal MPC controller.
Since for the double integrator example, on-line simulation does not
takes much time, both plotting and the simulation tasks are in 
the same script.
"""
import sys
sys.path.append('lib/')
from python_utils import PickleTool
from controller_evaluation import _simulate_scenarios

# Simulate all the scenarios.
_simulate_scenarios(plant_name='cstrs', 
                    controller_name='mpc',
                    Nsim=4320)
