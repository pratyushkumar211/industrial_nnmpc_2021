# [depends] cstrs_parameters.pickle cstrs_train_unstd.pickle
# [depends] %LIB%/controller_evaluation.py
# [depends] %LIB%/python_utils.py
# [depends] cstrs_comparision_plots.py
# [makes] pickle
"""
Script to run an on-line simulation for the 
CSTRs with flash example using the approximate NN controller.
"""
import sys
sys.path.append('lib/')
from python_utils import (PickleTool, figure_size_a4)
from matplotlib.backends.backend_pdf import PdfPages
from cstrs_comparision_plots import _cstrs_cl_comparision_plots
from controller_evaluation import _simulate_neural_network_unstd

# Load data and do an online simulation using the linear MPC controller.
(plants, controllers) = _simulate_neural_network_unstd(plant_name='cstrs', 
                                    Nsim=4320, nnwithuprev=True)

# Now make the plots here.
cstrs_parameters = PickleTool.load(filename='cstrs_parameters.pickle', 
                                        type='read')
scenarios = cstrs_parameters['online_test_scenarios']
plant_parameters = cstrs_parameters['cstrs_plant_parameters']

mpc_parameters = PickleTool.load(filename='cstrs_mpc.pickle', 
                                        type='read')
(mpc_plants, mpc_controllers) = (mpc_parameters['plants'], 
                                 mpc_parameters['controllers'])
figures = _cstrs_cl_comparision_plots(scenarios, 
                            mpc_plants, mpc_controllers,
                            plants, controllers, 
                            'NN-UNSTD',
                            plant_parameters,
                            (0, 4320))

# Create the Pdf figure object and save.
with PdfPages('cstrs_neural_network_unstd.pdf', 'w') as pdf_file:
    for fig in figures:
        pdf_file.savefig(fig)
