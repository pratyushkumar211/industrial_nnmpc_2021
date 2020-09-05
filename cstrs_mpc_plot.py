# [depends] cstrs_parameters.pickle cstrs_mpc.pickle
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
import itertools
from cstrs_labels import (ylabels, ulabels, zlabels, pdlabels)
import matplotlib.pyplot as plt
from python_utils import (PickleTool, figure_size_a4)
from matplotlib.backends.backend_pdf import PdfPages
from cstrs_comparision_plots import _get_cstr_data_for_plotting

def _cstrs_optimal_plot(plant, controller, parameters, 
                        setpoints, plot_range):
    """Data to plot.
    z: 4 outputs with setpoints. y:90 outputs. u:32 inputs.
    """
    # the x coordinate of the ylabels.
    ylabel_xcoordinate = -0.25
    # Get the arrays.
    (t, z, y, u, pest, p, 
      ulb, uub, zsp, ell) = _get_cstr_data_for_plotting(plant,  
                                                        controller, setpoints,
                                                        parameters, plot_range)
    # Make the plots.
    figures = plot_inputs(t, u, ulabels, ylabel_xcoordinate, ulb, uub)
    figures += plot_outputs(t, y, ylabels, ylabel_xcoordinate)
    figures += plot_controlled_outputs(t, z, zsp, zlabels, ylabel_xcoordinate)
    figures += plot_disturbances(t, p, pest, parameters['exp_dist_indices'], 
                                 pdlabels, ylabel_xcoordinate)
    figures += plot_average_stage_costs(t, ell, 
                            ylabel_xcoordinate)
    # Return the list of figures.
    return figures

def plot_inputs(t, u, ulabels, ylabel_xcoordinate, ulb, uub):
    """Return 4 figure objects which contain the plots of the inputs."""
    input_index = 0
    num_rows = 3
    num_cols = 2
    (figure, axes) = plt.subplots(nrows=num_rows, ncols=num_cols,
                                 sharex='col', figsize=figure_size_a4, 
                                 gridspec_kw = dict(wspace=0.5))
    for (row, column) in itertools.product(range(num_rows), range(num_cols)):
        axes[row, column].plot(t, u[:, input_index])
        axes[row, column].plot(t, ulb[:, input_index], 'k')
        axes[row, column].plot(t, uub[:, input_index], 'k')
        axes[row, column].set_ylabel(ulabels[input_index], rotation=False)
        axes[row, column].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        input_index += 1
    # Times.
    axes[2, 0].set_xlabel("Time (hr)")
    axes[2, 1].set_xlabel("Time (hr)")
    # Return the figure.
    return [figure]

def plot_controlled_outputs(t, z, zsp, ylabels, ylabel_xcoordinate):
    """Return 10 figure objects which contain the plots of the inputs."""
    output_index = 0
    num_rows = 3
    num_cols = 2
    (figure, axes) = plt.subplots(nrows=num_rows, ncols=num_cols,
                                  sharex='col', figsize=figure_size_a4,
                                  gridspec_kw = dict(wspace=0.5))
    for (row, column) in itertools.product(range(num_rows), range(num_cols)):
        axes[row, column].plot(t, z[:, output_index])
        axes[row, column].plot(t, zsp[:, output_index], 'r--')
        axes[row, column].set_xlim([np.min(t), np.max(t)])
        axes[row, column].set_ylabel(ylabels[output_index], rotation=False)
        axes[row, column].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        output_index += 1
    # Set the x axis label.
    axes[2, 0].set_xlabel("Time (hr)")
    axes[2, 1].set_xlabel("Time (hr)")
    # Return
    return [figure]

def plot_outputs(t, y, ylabels, ylabel_xcoordinate):
    """Return 1 figure objects which contain the plots of the inputs."""
    output_index = 0
    num_rows = 6
    num_cols = 2
    (figure, axes) = plt.subplots(nrows=num_rows, ncols=num_cols,
                                  sharex='col', figsize=figure_size_a4,
                                  gridspec_kw = dict(wspace=0.5))
    for (row, column) in itertools.product(range(num_rows), range(num_cols)):
        axes[row, column].plot(t, y[:, output_index])
        axes[row, column].set_ylabel(ylabels[output_index], rotation=False)
        axes[row, column].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        output_index += 1
    # Return
    return [figure]

def plot_disturbances(t, p, pest, exp_dist_indices, 
                      plabels, ylabel_xcoordinate):
    """Return a figure object which contain the plots of the inputs."""
    num_disturbances = p.shape[1]
    exp_dist_counter = 0
    (fig, axes) = plt.subplots(nrows=num_disturbances, ncols=1,
                               sharex=True, figsize=figure_size_a4)
    for disturbance_index in range(num_disturbances):
        axes[disturbance_index].plot(t, p[:, disturbance_index], 'b')
        if disturbance_index in exp_dist_indices:
            axes[disturbance_index].plot(t, pest[:, exp_dist_counter], 'r')
            exp_dist_counter +=1 
        axes[disturbance_index].set_ylabel(
            plabels[disturbance_index])
    fig.legend(labels = ('$p$', '$\hat{p}$'))
    return [fig]

def plot_average_stage_costs(t, ell, ylabel_xcoordinate):
    """Return a figure object which contain the plots of the inputs."""
    (figure, axes) = plt.subplots(nrows=1, ncols=1,
                               sharex=True, figsize=figure_size_a4)
    axes.plot(t, ell, 'b')
    axes.set_ylabel('$\Lambda_k$', rotation=False)
    return [figure]

# Load data and do an online simulation using the linear MPC controller.
cstrs_parameters = PickleTool.load(filename='cstrs_parameters.pickle', 
                                   type='read')
cstrs_online_test_scenarios = cstrs_parameters['online_test_scenarios']
cstrs_plant_parameters = cstrs_parameters['cstrs_plant_parameters']
cstrs_optimal = PickleTool.load(filename='cstrs_mpc.pickle', type='read')

# Make the plot.
figures = []
counter = 0
plot_start = 0
plot_end = 710
for scenario in cstrs_online_test_scenarios:

    (setpoints, _)  = scenario
    figures += _cstrs_optimal_plot(cstrs_optimal['plants'][counter], 
                                   cstrs_optimal['controllers'][counter],
                                   cstrs_plant_parameters, 
                                   setpoints, (plot_start, plot_end))
    counter+=1
# Create the Pdf figure object and save.
with PdfPages('cstrs_mpc_plot.pdf', 'w') as pdf_file:
    for fig in figures:
        pdf_file.savefig(fig)