# [depends] cdu_parameters.pickle cdu_train_unstd.pickle
# [makes] pickle
"""
Script to run an on-line simulation for the double integrator example
using the approximate NN controller.
"""
import sys
sys.path.append('lib/')
from python_utils import (PickleTool, figure_size_a4)
from matplotlib.backends.backend_pdf import PdfPages
from cdu_comparision_plots import _cdu_cl_comparision_plots
from controller_evaluation import _simulate_neural_network_unstd

# Load data and do an online simulation using the linear MPC controller.
(plants, controllers) = _simulate_neural_network_unstd(plant_name='cdu', 
                                    Nsim=2880, nnwithuprev=False)

# Load data and do an online simulation using the linear MPC controller.
cdu_parameters = PickleTool.load(filename='cdu_parameters.pickle', 
                                 type='read')
scenarios = cdu_parameters['online_test_scenarios']
plant_parameters = cdu_parameters['cdu_plant_parameters']

mpc_parameters = PickleTool.load(filename='cdu_mpc.pickle', 
                                        type='read')
(mpc_plants, mpc_controllers) = (mpc_parameters['plants'], 
                                 mpc_parameters['controllers'])
figures = _cdu_cl_comparision_plots(scenarios, 
                            mpc_plants, mpc_controllers,
                            plants, controllers, 
                            'NN-UNSTD',
                            plant_parameters,
                            (0, 2880))

# Save the figures for the two simulations.
with PdfPages('cdu_neural_network_unstd.pdf', 'w') as pdf_file:
    for fig in figures:
        pdf_file.savefig(fig)