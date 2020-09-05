# [depends] cdu_parameters.pickle cdu_optimal.pickle
"""
Script to run an on-line simulation using the double integrator example
using the optimal MPC controller.
Since for the double integrator example, on-line simulation does not
takes much time, both plotting and the simulation tasks are in
the same script.
"""
# import custompath
# custompath.add()
import sys
sys.path.append('lib/')
import numpy as np
import plottools
import itertools
from cdu_labels import ulabels, ylabels, zlabels, pdlabels
import matplotlib.pyplot as plt
from python_utils import (PickleTool, figure_size_a4)
from matplotlib.backends.backend_pdf import PdfPages

def _get_arrays_for_plotting(optimal, satdlqr, 
                             cdu_plotting_parameters, 
                             setpoints, plot_range):
    """ Get the appropriate arrays for plotting."""

    # Get some plotting cosmetics.
    yscale = cdu_plotting_parameters['yscale']
    uscale = cdu_plotting_parameters['uscale']
    lb = cdu_plotting_parameters['lb'] 
    ub = cdu_plotting_parameters['ub']
    us = cdu_plotting_parameters['us'] 
    ys = cdu_plotting_parameters['ys']
    disturbance_indices = cdu_plotting_parameters['disturbance_indices']

    (start, end) = plot_range
    #  Get the H matrix.
    (Ny, _) = optimal.C.shape
    Nc = 4
    H = np.concatenate((np.zeros((Nc, Ny-Nc)), np.eye(Nc)), axis=1)

    # Scaling for the inputs, outputs.
    pscale = np.take(uscale, disturbance_indices).squeeze()
    yscale = yscale.squeeze()
    uscale = uscale.squeeze()
    zscale = (H @ yscale[:, np.newaxis]).squeeze()
    
    # States.
    x_opt = np.asarray(optimal.x[start:end]).squeeze()
    x_satdlqr = np.asarray(satdlqr.x[start:end]).squeeze()

    # Control inputs.
    u_opt = np.asarray(optimal.u[start:end]).squeeze()*uscale + us.squeeze()
    u_satdlqr = np.asarray(satdlqr.u[start:end]).squeeze()*uscale + us.squeeze()

    # Disturbances.
    p = np.asarray(optimal.p[start:end]).squeeze()*pscale

    # All the outputs.
    y_opt = np.asarray(optimal.y[start:end]).squeeze()*yscale + ys.squeeze()
    y_satdlqr = np.asarray(satdlqr.y[start:end]).squeeze()*yscale + ys.squeeze()

    # The time index.
    t = np.asarray(optimal.t[start:end]).squeeze()

    # Outputs without offset.
    zs = H @ ys
    z_opt = np.asarray(optimal.y[start:end]).squeeze(axis=-1) @ H.T
    z_opt = z_opt*zscale+ zs.squeeze()
    z_satdlqr = np.asarray(satdlqr.y[start:end]).squeeze(axis=-1) @ H.T
    z_satdlqr = z_satdlqr*zscale + zs.squeeze()
    zsp = (setpoints[start:end, :] @ H.T)*zscale + zs.squeeze()

    # Upper and lower bounds on the control input.
    ulb = np.repeat(lb['u'].T, end-start, axis=0)*uscale + us.squeeze()
    uub = np.repeat(ub['u'].T, end-start, axis=0)*uscale + us.squeeze()

    # Return the arrays.
    return ((x_opt, x_satdlqr), (u_opt, u_satdlqr), p, (z_opt, z_satdlqr), 
             ulb, uub, (y_opt, y_satdlqr), zsp, t)

def _cdu_comparision_plot(optimal, satdlqr, 
                          cdu_plotting_parameters, setpoints, plot_range):
    """ Data to plot.
        z: 4 outputs with setpoints. y:90 outputs. u:32 inputs.
    """

    # The H matrix
    (Ny, _) = optimal.C.shape
    Nc = 4
    H = np.concatenate((np.zeros((Nc, Ny-Nc)), np.eye(Nc)), axis=1)

    # the x coordinate of the ylabels.
    ylabel_xcoordinate = -0.25
    
    # Optimal MPC arrays.
    (x, u, p, z, ulb, uub, y, z_sp, t) = _get_arrays_for_plotting(optimal, satdlqr, 
                                                                  cdu_plotting_parameters, 
                                                                  setpoints, plot_range)

    # Make the plots.
    figures = _plot_inputs(t, u, ulabels, ylabel_xcoordinate, ulb, uub, num_pages=4)
    figures += _plot_outputs(t, y, ylabels, ylabel_xcoordinate, num_pages=9)
    figures += _plot_controlled_outputs(t, z, z_sp, zlabels, ylabel_xcoordinate)
    figures += _plot_disturbances(t, p, pdlabels, ylabel_xcoordinate)

    # Return the list of figures
    return figures

def _plot_inputs(t, u, ulabels, ylabel_xcoordinate, ulb, uub, num_pages):
    """Return 4 figure objects which contain the plots of the inputs."""
    input_index = 0
    num_rows = 4
    num_cols = 2
    figs_and_axes = [plt.subplots(nrows=num_rows, ncols=num_cols,
                                  sharex='col', figsize=figure_size_a4, 
                                  gridspec_kw = dict(wspace=0.5))
                    for _ in range(num_pages)]
    # Extract the respective inputs.
    (u_opt, u_satdlqr) = u
    for (page, row, column) in itertools.product(range(num_pages),
                                                 range(num_rows), range(num_cols)):
        (figure, axes) = figs_and_axes[page]
        axes[row, column].plot(t, u_opt[:, input_index], 'b')
        axes[row, column].plot(t, u_satdlqr[:, input_index], 'g')
        axes[row, column].plot(t, ulb[:, input_index], 'k')
        axes[row, column].plot(t, uub[:, input_index], 'k')
        axes[row, column].set_ylabel(ulabels[input_index], rotation=False)
        axes[row, column].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        input_index += 1
        # Figure wide legend.
        figure.legend(labels = ('OPT', 'sat(Kx)'), loc = (0.45, 0.9))
    
    # Create a list of 4 figures to regturn.
    return_figures = [figs_and_axes[page][0]
                      for page in range(num_pages)]
    return return_figures

def _plot_outputs(t, y, ylabels, ylabel_xcoordinate, num_pages):
    """Return 10 figure objects which contain the plots of the inputs."""
    output_index = 0
    num_rows = 5
    num_cols = 2
    figs_and_axes = [plt.subplots(nrows=num_rows, ncols=num_cols,
                                  sharex='col', figsize=figure_size_a4,
                                  gridspec_kw = dict(wspace=0.5))
                     for _ in range(num_pages)]
    # Extract the respective oupts.
    (y_opt, y_satdlqr) = y
    for (page, row, column) in itertools.product(range(num_pages),
                                                 range(num_rows), range(num_cols)):
        (figure, axes) = figs_and_axes[page]
        axes[row, column].plot(t, y_opt[:, output_index], 'b')
        axes[row, column].plot(t, y_satdlqr[:, output_index], 'g')
        axes[row, column].set_ylabel(ylabels[output_index], rotation=False)
        axes[row, column].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        output_index += 1
        figure.legend(labels = ('OPT', 'sat(Kx)'), loc = (0.45, 0.9))

    return_figures = [figs_and_axes[page][0]
                      for page in range(num_pages)]
    return return_figures

def _plot_controlled_outputs(t, z, zsp, zlabels, ylabel_xcoordinate):
    """Return a figure object which contain the plots of the inputs."""
    # Extract the respective inputs.
    (z_opt, z_satdlqr) = z
    num_controlled_outputs = z_opt.shape[1]
    (figure, axes) = plt.subplots(nrows=num_controlled_outputs, ncols=1,
                               sharex=True, figsize=figure_size_a4)
    for output_index in range(num_controlled_outputs):
        axes[output_index].plot(t, z_opt[:, output_index], 'b')
        axes[output_index].plot(t, z_satdlqr[:, output_index], 'g')
        axes[output_index].plot(t, zsp[:, output_index], 'r--')
        axes[output_index].set_xlim([np.min(t), np.max(t)]) 
        axes[output_index].set_ylabel(zlabels[output_index], fontsize=14)
    
    figure.legend(labels = ('Online QP', 'sat(Kx)'), loc = (0.2, 0.9), ncol=2)
    axes[output_index].set_xlabel("Time (min)")
    return [figure]

def _plot_disturbances(t, p, plabels, ylabel_xcoordinate):
    """Return a figure object which contain the plots of the inputs."""
    num_disturbances = p.shape[1]
    (fig, axes) = plt.subplots(nrows=num_disturbances, ncols=1,
                               sharex=True, figsize=figure_size_a4)
    for disturbance_index in range(num_disturbances):
        axes[disturbance_index].plot(t, p[:, disturbance_index])
        axes[disturbance_index].set_ylabel(
            plabels[disturbance_index])
    return [fig]

# Load data and do an online simulation using the linear MPC controller.
cdu_parameters = PickleTool.load(filename='cdu_parameters.pickle', type='read')
online_test_scenarios = cdu_parameters['online_test_scenarios']
cdu_plotting_parameters = cdu_parameters['cdu_plotting_parameters']

# Get the optimal and the satdlqr pickle files.
cdu_optimal = PickleTool.load(filename='cdu_optimal.pickle', type='read')
cdu_satdlqr = PickleTool.load(filename='cdu_satdlqr.pickle', type='read')
optimal_plants = cdu_optimal['plants']
optimal_controllers = cdu_optimal['controllers']
satdlqr_plants = cdu_satdlqr['plants']
satdlqr_controllers = cdu_satdlqr['controllers']

# Make the plot.
figures = []
plot_start = 0
plot_end = 3600
counter = 0
# Get the arrays to store the metrics.
satdlqr_metric = np.zeros((len(online_test_scenarios), 1))
optimal_metric = np.zeros((len(online_test_scenarios), 1))

for scenario in online_test_scenarios:

    # The scenario.
    (setpoints, _)  = online_test_scenarios[counter]

    # First make plots of the closed-loop trajectories.
    figures += _cdu_comparision_plot(optimal_plants[counter], 
                                     satdlqr_plants[counter],
                                     cdu_plotting_parameters,
                                     setpoints,
                                     (plot_start, plot_end))

    # Get the performance metrics in one place.
    optimal_metric[counter, 0] = optimal_controllers[counter].average_stage_costs[-1].squeeze()
    satdlqr_metric[counter, 0] = satdlqr_controllers[counter].average_stage_costs[-1].squeeze()
    counter+=1

performance_loss = 100*(satdlqr_metric - optimal_metric)/optimal_metric
# Create the Pdf figure object and save.
with PdfPages('cdu_satdlqr_plot.pdf', 'w') as pdf_file:
    for fig in figures:
        pdf_file.savefig(fig)