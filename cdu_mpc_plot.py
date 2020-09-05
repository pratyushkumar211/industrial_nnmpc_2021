# [depends] cdu_parameters.pickle cdu_mpc.pickle
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
from cdu_labels import (ylabels, ulabels, zlabels, pdlabels)
import matplotlib.pyplot as plt
from python_utils import (PickleTool, figure_size_a4)
from matplotlib.backends.backend_pdf import PdfPages
from cdu_comparision_plots import _get_cdu_data_for_plotting

def _cdu_optimal_plot(plant, controller, parameters, 
                        setpoints, plot_range):
    """Data to plot.
    z: 4 outputs with setpoints. y:90 outputs. u:32 inputs.
    """
    # the x coordinate of the ylabels.
    ylabel_xcoordinate = -0.25
    # Get the arrays.
    (t, z, y, u, pest, p, 
      ulb, uub, zsp, ell) = _get_cdu_data_for_plotting(plant,  
                                                       controller, setpoints,
                                                       parameters, plot_range)
    # Make the plots.
    figures = plot_inputs(t, u, ulabels, ylabel_xcoordinate, ulb, uub)
    figures += plot_outputs(t, y, ylabels, ylabel_xcoordinate)
    figures += plot_controlled_outputs(t, z, zsp, zlabels, ylabel_xcoordinate)
    figures += plot_disturbances(t, p, pest, 
                                 pdlabels, ylabel_xcoordinate)
    figures += plot_average_stage_costs(t, ell, 
                            ylabel_xcoordinate)
    # Return the list of figures.
    return figures

def plot_inputs(t, u, ulabels, ylabel_xcoordinate, 
                ulb, uub):
    """Return 4 figure objects which contain the plots of the inputs."""
    num_pages = 2
    num_rows = 8
    num_cols = 2
    input_index = 0
    figs_and_axes = [plt.subplots(nrows=num_rows, ncols=num_cols,
                                 sharex='col', figsize=figure_size_a4, 
                                 gridspec_kw = dict(wspace=0.5))
                            for _ in range(num_pages)]
    for (page, row, column) in itertools.product(range(num_pages),
                                                 range(num_rows), 
                                                 range(num_cols)):
        (figure, axes) = figs_and_axes[page]
        axes[row, column].plot(t, u[:, input_index], 'b')
        axes[row, column].plot(t, ulb[:, input_index], 'k')
        axes[row, column].plot(t, uub[:, input_index], 'k')
        axes[row, column].set_ylabel(ulabels[input_index], rotation=False)
        axes[row, column].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        input_index += 1
        # x-axis label.
        axes[7, 0].set_xlabel("Time (hr)")
        axes[7, 1].set_xlabel("Time (hr)")

    # Get the figures to return.
    return_figures = [figs_and_axes[page][0]
                      for page in range(num_pages)]
    # Return the figure.
    return return_figures

def plot_controlled_outputs(t, z, zsp, zlabels, ylabel_xcoordinate):
    """Return 10 figure objects which contain the plots of the inputs."""
    output_index = 0
    num_rows = z.shape[1]
    num_cols = 1
    (figure, axes) = plt.subplots(nrows=num_rows, ncols=num_cols,
                                  squeeze=False,
                                  sharex='col', figsize=figure_size_a4,
                                  gridspec_kw = dict(wspace=0.5))
    for (row, column) in itertools.product(range(num_rows), range(num_cols)):
        axes[row, column].plot(t, z[:, output_index])
        axes[row, column].plot(t, zsp[:, output_index], 'r--')
        axes[row, column].set_xlim([np.min(t), np.max(t)])
        axes[row, column].set_ylabel(zlabels[output_index], rotation=False)
        axes[row, column].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        output_index += 1
    # Set the x axis label.
    axes[3, 0].set_xlabel("Time (hr)")
    # Return
    return [figure]

def plot_outputs(t, y, ylabels, 
                 ylabel_xcoordinate):
    """Return 1 figure objects which contain the plots of the inputs."""
    num_pages = 5
    num_rows = 9
    num_cols = 2
    output_index = 0
    num_outputs = y.shape[1]
    figs_and_axes = [plt.subplots(nrows=num_rows, ncols=num_cols,
                                  sharex='col', figsize=figure_size_a4,
                                  gridspec_kw = dict(wspace=0.5))
                            for _ in range(num_pages)]
    for (page, row, column) in itertools.product(range(num_pages),
                                                 range(num_rows), 
                                                 range(num_cols)):
        (figure, axes) = figs_and_axes[page]
        if output_index < num_outputs:
            axes[row, column].plot(t, y[:, output_index], 'b')
            axes[row, column].set_ylabel(ylabels[output_index], rotation=False)
            yaxis = axes[row, column].get_yaxis()
            yaxis.set_label_coords(ylabel_xcoordinate, 0.5) 
            output_index += 1
        # Get the x-axis label and figure title.
        axes[8, 0].set_xlabel("Time (hr)")
        axes[8, 1].set_xlabel("Time (hr)")
    # Get the figures to return.
    return_figures = [figs_and_axes[page][0]
                      for page in range(num_pages)]
    # Return the figure.
    return return_figures

def plot_disturbances(t, p, pest, 
                      plabels, ylabel_xcoordinate):
    """Return a figure object which contain the plots of the inputs."""
    num_pages = 1
    num_rows = p.shape[1]
    num_cols = 1
    dist_counter = 0
    figs_and_axes = [plt.subplots(nrows=num_rows, ncols=num_cols,
                                  squeeze=False, 
                                  sharex=True, figsize=figure_size_a4)
                            for _ in range(num_pages)]
    for (page, row, col) in itertools.product(range(num_pages),
                                              range(num_rows),
                                              range(num_cols)):
        (figure, axes) = figs_and_axes[page]
        axes[row, col].plot(t, p[:, dist_counter], 'k')
        axes[row, col].plot(t, pest[:, dist_counter], 'b')
        axes[row, col].set_ylabel(plabels[dist_counter])
        dist_counter +=1
        # Set labels and return.
        figure.legend(labels = ('$p$', '$\hat{p}^{MPC}$'), 
                      loc = (0.45, 0.9))
    # Get the figures to return.
    return_figures = [figs_and_axes[page][0]
                      for page in range(num_pages)]
    # Return the figure.
    return return_figures

def plot_average_stage_costs(t, ell, ylabel_xcoordinate):
    """Return a figure object which contain the plots of the inputs."""
    (figure, axes) = plt.subplots(nrows=1, ncols=1,
                               sharex=True, figsize=figure_size_a4)
    axes.plot(t, ell, 'b')
    axes.set_ylabel('$\ell_k$')
    return [figure]

# Load data and do an online simulation using the linear MPC controller.
cdu_parameters = PickleTool.load(filename='cdu_parameters.pickle', 
                                   type='read')
cdu_online_test_scenarios = cdu_parameters['online_test_scenarios']
cdu_plant_parameters = cdu_parameters['cdu_plant_parameters']
cdu_optimal = PickleTool.load(filename='cdu_mpc.pickle', type='read')

# Make the plot.
figures = []
counter = 0
plot_start = 0
plot_end = 1440
for scenario in cdu_online_test_scenarios:

    (setpoints, _)  = scenario
    figures += _cdu_optimal_plot(cdu_optimal['plants'][counter], 
                                 cdu_optimal['controllers'][counter],
                                 cdu_plant_parameters, 
                                 setpoints, (plot_start, plot_end))
    counter+=1
# Create the Pdf figure object and save.
with PdfPages('cdu_mpc_plot.pdf', 'w') as pdf_file:
    for fig in figures:
        pdf_file.savefig(fig)