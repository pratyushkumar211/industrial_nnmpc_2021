# [depends] cdu_parameters.pickle cdu_mpc.pickle cdu_us.pickle
# [depends] cdu_satdlqr.pickle cdu_short_horizon.pickle cdu_train.pickle
# [depends] cdu_neural_network.pickle
# [depends] %LIB%/python_utils.py
# [depends] %LIB%/cdu_labels.py
# [depends] %LIB%/controller_evaluation.py
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
import plottools
import numpy as np
import itertools
import matplotlib.pyplot as plt
from cdu_labels import (ulabels, zlabels, ylabels, pdlabels)
from python_utils import (PickleTool, figure_size_cl,
                         figure_size_a4, figure_size_metrics)
from matplotlib.backends.backend_pdf import PdfPages
from controller_evaluation import (plot_cl_performance_and_comp_times,
                                   plot_nn_vs_num_samples_performance,
                                   _load_data_for_plots, _set_font_size,
                                   _get_best_nn_plant_controllers)

def _get_cdu_data_for_plotting(plant, controller, setpoints,
                               parameters, plot_range):
    """ Get the appropriate arrays for plotting."""
    # Get some plotting cosmetics.
    yscale = parameters['yscale'].squeeze(axis=-1)
    uscale = parameters['uscale'].squeeze(axis=-1)
    ulb = parameters['lb']['u']
    uub = parameters['ub']['u']
    us = parameters['us'].squeeze(axis=-1) 
    ys = parameters['ys'].squeeze(axis=-1)
    dist_indices = parameters['dist_indices']
    pscale = uscale[dist_indices, ]
    (start, end) = plot_range
    #  Get the H matrix.
    (Ny, _) = plant.C.shape
    Nc = 4
    H = np.concatenate((np.zeros((Nc, Ny-Nc)), np.eye(Nc)), axis=1)
    # Get the plant arrays.
    y = np.asarray(plant.y[start:end]).squeeze()*yscale + ys
    z = y @ H.T
    u = np.asarray(plant.u[start:end]).squeeze()*uscale + us
    p = np.asarray(plant.p[start:end]).squeeze()#*pscale
    t = np.asarray(plant.t[start:end]).squeeze()/60
    # Get the controller arrays.
    pest = np.asarray(controller.target_selector.dhats[start:end]).squeeze()
    #pest = pest*pscale
    ell = np.asarray(controller.average_stage_costs[start:end]).squeeze()
    # Get the setpoints.
    ysp = setpoints[start:end, :]*yscale + ys
    zsp = ysp @ H.T
    # Upper and lower bounds on the control input.
    ulb = np.repeat(ulb.T, end-start, axis=0)*uscale + us
    uub = np.repeat(uub.T, end-start, axis=0)*uscale + us
    # Return the arrays.
    return (t, z, y, u, pest, p, ulb, uub, zsp, ell)

def _cdu_cl_comparision_plots(scenarios, 
                             mpc_plants, mpc_controllers,
                             fast_plants, fast_controllers, 
                             fast_controller_name,
                             parameters,
                             plot_range):
    """Data to plot.
    z: 4 outputs with setpoints. y:90 outputs. u:32 inputs.
    """
    # the x coordinate of the ylabels.
    ylabel_xcoordinate = -0.08
    figures = []
    for (scenario, mpc_plant, 
         mpc_controller, fast_plant,
         fast_controller) in zip(scenarios, mpc_plants, 
                                 mpc_controllers, fast_plants,
                                 fast_controllers):
        (setpoints, _) = scenario
        # Get the arrays.
        (t, z_mpc, y_mpc, u_mpc, pest_mpc, 
        p, ulb, uub, zsp, ell_mpc) = _get_cdu_data_for_plotting(mpc_plant, 
                                                  mpc_controller,
                                                  setpoints, 
                                                  parameters, 
                                                  plot_range)
        (t, z_fast, y_fast, u_fast, pest_fast, 
        p, _, _, _, ell_fast) = _get_cdu_data_for_plotting(fast_plant, 
                                                  fast_controller,
                                                  setpoints, 
                                                  parameters, 
                                                  plot_range)
        # Make the Plots.
        figures += plot_inputs(t, (u_mpc, u_fast), ulabels, 
                              ylabel_xcoordinate, ulb, uub, fast_controller_name)
        figures += plot_outputs(t, (y_mpc, y_fast), 
                                ylabels, ylabel_xcoordinate,
                                fast_controller_name)
        figures += plot_controlled_outputs(t, (z_mpc, z_fast), zsp, 
                                zlabels, ylabel_xcoordinate,
                                fast_controller_name)
        figures += plot_disturbances(t, (pest_mpc, pest_fast), p, 
                                     pdlabels, 
                                     ylabel_xcoordinate,
                                     fast_controller_name)
        figures += plot_average_stage_costs(t, (ell_mpc, ell_fast), 
                            ylabel_xcoordinate,
                            fast_controller_name)
    # Return the list of figures.
    return figures

def plot_inputs(t, u, ulabels, ylabel_xcoordinate, 
                ulb, uub, fast_controller_name):
    """Return 4 figure objects which contain the plots of the inputs."""
    (u_mpc, u_fast) = u
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
        axes[row, column].plot(t, u_mpc[:, input_index], 'b')
        axes[row, column].plot(t, u_fast[:, input_index], 'g')
        axes[row, column].plot(t, ulb[:, input_index], 'k')
        axes[row, column].plot(t, uub[:, input_index], 'k')
        axes[row, column].set_ylabel(ulabels[input_index], rotation=False)
        axes[row, column].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        input_index += 1
        # x-axis label.
        axes[num_rows-1, 0].set_xlabel("Time (hr)")
        axes[num_rows-1, 1].set_xlabel("Time (hr)")
        figure.legend(labels = ('MPC', fast_controller_name), loc = (0.45, 0.9))
    # Get the figures to return.
    return_figures = [figs_and_axes[page][0]
                      for page in range(num_pages)]
    # Return the figure.
    return return_figures

def plot_outputs(t, y, ylabels, 
                 ylabel_xcoordinate, fast_controller_name):
    """Return 1 figure objects which contain the plots of the inputs."""
    (y_mpc, y_fast) = y
    num_pages = 5
    num_rows = 9
    num_cols = 2
    output_index = 0
    figs_and_axes = [plt.subplots(nrows=num_rows, ncols=num_cols,
                                  squeeze=False,
                                  sharex='col', figsize=figure_size_a4,
                                  gridspec_kw = dict(wspace=0.5))
                            for _ in range(num_pages)]
    for (page, row, column) in itertools.product(range(num_pages),
                                                 range(num_rows), 
                                                 range(num_cols)):
        (figure, axes) = figs_and_axes[page]
        axes[row, column].plot(t, y_mpc[:, output_index], 'b')
        axes[row, column].plot(t, y_fast[:, output_index], 'g'),
        axes[row, column].set_ylabel(ylabels[output_index], rotation=False)
        yaxis = axes[row, column].get_yaxis()
        yaxis.set_label_coords(ylabel_xcoordinate, 0.5) 
        output_index += 1
        # Get the x-axis label and figure title.
        axes[num_rows-1, column].set_xlabel("Time (hr)")
        figure.legend(labels = ('MPC', fast_controller_name), 
                      loc = (0.3, 0.9), ncol=2)
    # Get the figures to return.
    return_figures = [figs_and_axes[page][0]
                      for page in range(num_pages)]
    # Return the figure.
    return return_figures

def plot_controlled_outputs(t, z, zsp, ylabels, 
                 ylabel_xcoordinate, fast_controller_name):
    """Return 1 figure objects which contain the plots of the inputs."""
    (z_mpc, z_fast) = z
    num_pages = 1
    num_rows = z_mpc.shape[1]
    num_cols = 1
    output_index = 0
    figs_and_axes = [plt.subplots(nrows=num_rows, ncols=num_cols,
                                  squeeze=False,
                                  sharex='col', figsize=figure_size_a4,
                                  gridspec_kw = dict(wspace=0.5))
                            for _ in range(num_pages)]
    for (page, row, column) in itertools.product(range(num_pages),
                                                 range(num_rows), 
                                                 range(num_cols)):
        (figure, axes) = figs_and_axes[page]
        axes[row, column].plot(t, z_mpc[:, output_index], 'b')
        axes[row, column].plot(t, z_fast[:, output_index], 'g')
        axes[row, column].plot(t, zsp[:, output_index], 'r--')
        axes[row, column].set_ylabel(zlabels[output_index])
        axes[row, column].set_xlim([np.min(t), np.max(t)])
        yaxis = axes[row, column].get_yaxis()
        yaxis.set_label_coords(ylabel_xcoordinate, 0.5) 
        output_index += 1
        # Get the x-axis label and figure title.
        axes[num_rows-1, 0].set_xlabel("Time (hr)")
        figure.legend(labels = ('MPC', fast_controller_name), 
                      loc = (0.3, 0.9), ncol=2)
    # Get the figures to return.
    return_figures = [figs_and_axes[page][0]
                      for page in range(num_pages)]
    # Return the figure.
    return return_figures

def plot_disturbances(t, pest, p, 
                      plabels, ylabel_xcoordinate,
                      fast_controller_name):
    """Return a figure object which contain the plots of the inputs."""
    (pest_mpc, pest_fast) = pest
    num_pages = 1
    num_rows = p.shape[1]
    num_cols = 1
    dist_counter = 0
    figs_and_axes = [plt.subplots(nrows=num_rows, ncols=num_cols,
                                  squeeze=False, sharex=True, 
                                  figsize=figure_size_a4)
                            for _ in range(num_pages)]
    for (page, row, col) in itertools.product(range(num_pages),
                                              range(num_rows),
                                              range(num_cols)):
        (figure, axes) = figs_and_axes[page]
        axes[row, col].plot(t, p[:, dist_counter], 'k')
        axes[row, col].plot(t, pest_mpc[:, dist_counter], 'b')
        axes[row, col].plot(t, pest_fast[:, dist_counter], 'g')
        axes[row, col].set_ylabel(plabels[dist_counter])
        dist_counter +=1
        # Set labels and return.
        figure.legend(labels = ('$p$', '$\hat{p}^{MPC}$', 
                 '$\hat{p}^{' + fast_controller_name + '}$'), 
                      loc = (0.45, 0.9))
    # Get the figures to return.
    return_figures = [figs_and_axes[page][0]
                      for page in range(num_pages)]
    # Return the figure.
    return return_figures

def plot_average_stage_costs(t, ell, 
                            ylabel_xcoordinate,
                            fast_controller_name):
    """Return a figure object which contain the plots of the inputs."""
    (ell_mpc, ell_fast) = ell
    (figure, axes) = plt.subplots(nrows=1, ncols=1,
                               sharex=True, figsize=figure_size_a4)
    axes.plot(t, ell_mpc, 'b')
    axes.plot(t, ell_fast, 'g')
    axes.set_ylabel('$\ell_k$')
    figure.legend(labels = ('MPC', fast_controller_name), loc = (0.45, 0.9))
    return [figure]

def main():
    """ Run these commands when this script is called."""

    # Set the fontize.
    _set_font_size(paper_fontsize=16)

    # Get the parameters and the plant object which has the closed-loop data
    # for the two controllers.
    (cdu_parameters, cdu_mpc, cdu_us, 
     cdu_satdlqr, cdu_short_horizon, 
     cdu_train, 
     cdu_neural_network) = _load_data_for_plots(plant_name='cdu')

    # Optimal/NN plants/controllers. 
    mpc_plants = cdu_mpc['plants']
    mpc_controllers = cdu_mpc['controllers']
    us_controllers = cdu_us['controllers']
    satdlqr_controllers = cdu_satdlqr['controllers']
    short_horizon_controllers = cdu_short_horizon['controllers']
    nn_plants = cdu_neural_network['plants']
    nn_controllers = cdu_neural_network['controllers']
    performance_loss = cdu_neural_network['performance_loss']

    # Number of architectures, NNs, scenarios.
    num_architectures = len(cdu_train['regulator_dims'])
    num_nns_per_architecture = len(cdu_train['num_samples'])
    num_scenarios = len(cdu_parameters['online_test_scenarios'])

    # Create lists to store the figures and the NN metrics.
    nn_labels = ['NN-3-1664', 'NN-3-1792', 'NN-3-1920', 'NN-3-2048']
    cl_figures = []
    scenarios = cdu_parameters['online_test_scenarios']
    (best_nn_plants, 
     best_nn_controllers) = _get_best_nn_plant_controllers(nn_plants, 
                                              nn_controllers,
                                              performance_loss,
                                              num_architectures, 
                                              num_nns_per_architecture,
                                              num_scenarios)
    for arch in range(num_architectures):
        # Get the NN plant amd controller object.
        fast_plants = [best_nn_plants.pop(0) for _ in range(num_scenarios)]
        fast_controllers = [best_nn_controllers.pop(0)
                            for _ in range(num_scenarios)]
        # First make plots of the closed-loop trajectories.
        cl_figures += _cdu_cl_comparision_plots(scenarios, 
                                 mpc_plants, mpc_controllers,
                                 fast_plants, fast_controllers, 
                                 nn_labels[arch],
                                 cdu_parameters['cdu_plant_parameters'],
                                 (60, 1440))

    # Make the plots for the closed loop metrics of NNs with 
    # increasing data.
    performance_loss = cdu_neural_network['performance_loss']
    num_samples = cdu_train['num_samples']
    cl_metric_figures = plot_nn_vs_num_samples_performance(performance_loss,
                                             num_samples, 
                                             nn_labels,
                                             figure_size_metrics,
                                             right_frac=0.9,
                                             legend_title_location=(0.6, 0.5),
                                             ylabel_xcoordinate=-0.1,
                                             yaxis_log=False)

    # Plot the average stage cost in time.
    time = np.asarray(cdu_mpc['plants'][0].t[0:-1]).squeeze()
    nn_controllers = PickleTool.load(filename='cdu_neural_network.pickle', 
                                     type='read')['controllers']
    cl_ct_figures = plot_cl_performance_and_comp_times(time,
                         mpc_controllers,
                         us_controllers, satdlqr_controllers,
                         short_horizon_controllers, 
                         nn_plants, nn_controllers, performance_loss,
                         num_architectures, num_nns_per_architecture, 
                         cl_xlabel = 'Time (min)', 
                         cl_legend_labels = ['MPC', 'satK'] + nn_labels,
                         cl_right_frac = 0.9,
                         ct_right_frac = 0.9, 
                         ct_legend_labels = ['MPC'] + nn_labels,
                         figure_size=figure_size_metrics,
                         fig_cl_title_location=(0.35, 0.55),
                         fig_ct_title_location='center',
                         cl_us=False,
                         cl_short_horizon=False,
                         cl_yaxis_log=False,
                         ylabel_xcoordinate=-0.1,
                         num_bins=2000)

    # Stuff all the figures into one.
    figures = cl_figures + cl_metric_figures + cl_ct_figures
    with PdfPages('cdu_comparision_plots.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Call the main function.
main()