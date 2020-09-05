# [depends] cstrs_offline_data.pickle
"""
Script to run an on-line simulation using the double integrator example
using the optimal MPC controller.
Since for the double integrator example, on-line simulation does not
takes much time, both plotting and the simulation tasks are in 
the same script.
"""
import sys
sys.path.append('../lib/')
import numpy as np
from cstrs_optimal_plot import (plot_inputs, plot_states, 
                                plot_outputs, plot_disturbances)
from cstrs_labels import (xlabels, ylabels, ulabels, pdlabels)
import matplotlib.pyplot as plt
from python_utils import (PickleTool, figure_size_a4)
from matplotlib.backends.backend_pdf import PdfPages

def _get_arrays_for_plotting(raw_data, num_samples):
    """ Convert lists to numpy array."""
    # Get the data arrays for plotting.
    x = raw_data['x'][:num_samples, :].squeeze()
    u = raw_data['useq'][:num_samples, 0:6].squeeze()
    d = raw_data['d'][:num_samples, :].squeeze()
    ysp = raw_data['rsp'][:num_samples, :].squeeze()
    y = raw_data['y'][:num_samples, :].squeeze()
    t = np.arange(0, num_samples)
    ulb = np.tile(np.NaN, (num_samples, u.shape[1]))
    uub = np.tile(np.NaN, (num_samples, u.shape[1]))
    return (x, u, d, ysp, y, t, ulb, uub)

def _cstrs_plot_offline_data(offline_simulator, num_samples):
    """ Get the offline data for plotting."""
    ylabel_xcoordinate = -0.25
    (x, u, d, ysp, y, t, ulb, uub) = _get_arrays_for_plotting(offline_simulator, num_samples)

    # Make the Plots.
    figures = plot_inputs(t, u, ulabels, ylabel_xcoordinate, ulb, uub)
    figures += plot_outputs(t, y, ysp, ylabels, ylabel_xcoordinate)
    figures += plot_states(t, x, xlabels, ylabel_xcoordinate)
    figures += plot_disturbances(t, d, pdlabels, ylabel_xcoordinate)
    return figures

def main():
    """ The Main Function. """
    # Load data and do an online simulation using the linear MPC controller.
    cstrs_offline_data = PickleTool.load(filename='cstrs_offline_data.pickle', type='read')
    # Make the plot. 
    figures = _cstrs_plot_offline_data(cstrs_offline_data, 
                                       num_samples=1200)

    # Create the Pdf figure object and save.
    with PdfPages('cstrs_offline_data.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

if __name__ == "__main__":
    main()
