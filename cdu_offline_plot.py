# [depends] cdu_parameters.pickle cdu_offline_data.pickle
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
from cdu_labels import ulabels, ylabels, zlabels, pdlabels
import matplotlib.pyplot as plt
from python_utils import (PickleTool, figure_size_a4)
from matplotlib.backends.backend_pdf import PdfPages

def _get_arrays_for_plotting(offline_data, parameters, num_samples):
    """ Get the appropriate arrays for plotting."""

    # Extract some parameters.
    setpoints = parameters['offline_simulator'].setpoints
    lb = parameters['cdu_plotting_parameters']['lb'] 
    ub = parameters['cdu_plotting_parameters']['ub']
    us = parameters['cdu_plotting_parameters']['us'] 
    ys = parameters['cdu_plotting_parameters']['ys']
    disturbance_indices = parameters['cdu_plotting_parameters']['disturbance_indices']

    #  Get the H matrix.
    (Ny, _) = parameters['plant'].C.shape
    Nc = 4
    (Nx, Nu) = parameters['plant'].B.shape
    H = np.concatenate((np.zeros((Nc, Ny-Nc)), np.eye(Nc)), axis=1)

    # Scaling for the inputs, outputs.
    yscale = parameters['cdu_plotting_parameters']['yscale'].squeeze()
    uscale = parameters['cdu_plotting_parameters']['uscale'].squeeze()
    pscale = np.take(uscale, disturbance_indices).squeeze()
    zscale = H @ yscale
    
    # Control inputs.
    u = np.asarray(offline_data['useq'][0:num_samples, ...]).squeeze(axis=-1)[:, 0:Nu]*uscale + us.squeeze()

    # Disturbances.
    p = np.asarray(offline_data['d'][0:num_samples, ...])*pscale

    # All the outputs.
    y = np.asarray(offline_data['y'][0:num_samples, ...]).squeeze(axis=-1)*yscale + ys.squeeze()

    # The time index.
    t = np.arange(num_samples)

    # Outputs without offset.
    zs = H @ ys
    z = np.asarray(offline_data['z'][0:num_samples, ...]).squeeze(axis=-1)*zscale + zs.squeeze()
    zsp = (setpoints[0:num_samples, :] @ H.T)*zscale + zs.squeeze()

    # Upper and lower bounds on the control input.
    ulb = np.repeat(lb['u'].T, num_samples, axis=0)*uscale + us.squeeze()
    uub = np.repeat(ub['u'].T, num_samples, axis=0)*uscale + us.squeeze()

    # Return the arrays.
    return (u, p, z, zsp, ulb, uub, y, t)

def _cdu_offline_plot(offline_data, parameters, num_samples):
    """Data to plot.
    z: 4 outputs with setpoints. y:90 outputs. u:32 inputs.
    """

    # the x coordinate of the ylabels.
    ylabel_xcoordinate = -0.25

    (u, p, z, zsp, ulb, uub, y, t) = _get_arrays_for_plotting(offline_data, parameters, num_samples)

    # Labels.
    figures = _plot_inputs(t, u, ulabels, ylabel_xcoordinate, ulb, uub, num_pages=4)
    figures += _plot_outputs(t, y, ylabels, ylabel_xcoordinate, num_pages=9)
    figures += _plot_controlled_outputs(t, z, zsp, zlabels, ylabel_xcoordinate)
    figures += _plot_disturbances(t, p, pdlabels, ylabel_xcoordinate)

    # Return the list of figures.
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
    for (page, row, column) in itertools.product(range(num_pages),
                                                 range(num_rows), range(num_cols)):
        (figure, axes) = figs_and_axes[page]
        axes[row, column].plot(t, u[:, input_index])
        axes[row, column].plot(t, ulb[:, input_index], 'k')
        axes[row, column].plot(t, uub[:, input_index], 'k')
        axes[row, column].set_ylabel(ulabels[input_index], rotation=False)
        axes[row, column].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        input_index += 1
    
    # Create a list of 4 figures to regturn
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
    for (page, row, column) in itertools.product(range(num_pages),
                                                 range(num_rows), range(num_cols)):
        (figure, axes) = figs_and_axes[page]
        axes[row, column].plot(t, y[:, output_index])
        axes[row, column].set_ylabel(ylabels[output_index], rotation=False)
        axes[row, column].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        output_index += 1

    return_figures = [figs_and_axes[page][0]
                      for page in range(num_pages)]
    return return_figures


def _plot_controlled_outputs(t, z, zsp, zlabels, ylabel_xcoordinate):
    """Return a figure object which contain the plots of the inputs."""
    num_controlled_outputs = z.shape[1]
    (fig, axes) = plt.subplots(nrows=num_controlled_outputs, ncols=1,
                               sharex=True, figsize=figure_size_a4)
    for output_index in range(num_controlled_outputs):
        axes[output_index].plot(t, z[:, output_index])
        axes[output_index].plot(t, zsp[:, output_index], 'r--')
        axes[output_index].set_ylabel(
            zlabels[output_index])
    return [fig]


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
cdu_offline_data = PickleTool.load(filename='cdu_offline_data.pickle', type='read')
cdu_parameters = PickleTool.load(filename='cdu_parameters.pickle', type='read')

# Make the plot.
figures = _cdu_offline_plot(cdu_offline_data, cdu_parameters, num_samples=1440)

# Create the Pdf figure object and save.
with PdfPages('cdu_offline_plot.pdf', 'w') as pdf_file:
    for fig in figures:
        pdf_file.savefig(fig)
