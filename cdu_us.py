# [depends] cdu_parameters.pickle
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
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from python_utils import PickleTool
from controller_evaluation import _simulate_scenarios
from cdu_comparision_plots import _cdu_cl_comparision_plots

# Simulate all the scenarios.
(plants, controllers) = _simulate_scenarios(plant_name='cdu', 
                                            controller_name='us', 
                                            Nsim=2880)

# Now make the plots here.
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
                            'us',
                            plant_parameters,
                            (0, 2880))
# Plot.
# Create the Pdf figure object and save.
with PdfPages('cdu_us.pdf', 'w') as pdf_file:
    for fig in figures:
        pdf_file.savefig(fig)