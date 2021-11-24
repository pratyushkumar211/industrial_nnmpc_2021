# [depends] cstrs_parameters.pickle
# [depends] %LIB%/controller_evaluation.py
# [depends] %LIB%/python_utils.py
# [depends] cstrs_comparision_plots.py
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
from cstrs_comparision_plots import _cstrs_cl_comparision_plots

# Simulate all the scenarios.
(plants, controllers) = _simulate_scenarios(plant_name='cstrs', 
                    controller_name='us', 
                    Nsim=4320)

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
                            'us',
                            plant_parameters,
                            (0, 4320))
# Plot.
# Create the Pdf figure object and save.
with PdfPages('cstrs_us.pdf', 'w') as pdf_file:
    for fig in figures:
        pdf_file.savefig(fig)
