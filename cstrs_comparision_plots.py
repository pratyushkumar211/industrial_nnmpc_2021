# [depends] cstrs_parameters.pickle cstrs_mpc.pickle cstrs_us.pickle
# [depends] cstrs_satdlqr.pickle cstrs_short_horizon.pickle cstrs_train.pickle
# [depends] cstrs_neural_network.pickle
# [depends] %LIB%/controller_evaluation.py
# [depends] %LIB%/cstrs_labels.py
""" 
Script to make plots for comparing the 
performance of the optimal MPC controller
and the NN MPC controller.
"""
import sys
sys.path.append('lib/')
import itertools
import numpy as np
import matplotlib.pyplot as plt
from cstrs_labels import (ylabels, zlabels, ulabels, pdlabels)
from python_utils import (PickleTool, figure_size_cl, figure_size_metrics)
from matplotlib.backends.backend_pdf import PdfPages
from controller_evaluation import (plot_cl_performance_and_comp_times,
                        plot_nn_vs_num_samples_performance,
                        _load_data_for_plots,
                        _get_best_nn_plant_controllers)

def _get_cstr_data_for_plotting(plant, controller, setpoints,
                                parameters, plot_range):
    """ Get the appropriate arrays for plotting."""
    # Get some plotting cosmetics.
    yscale = parameters['yscale']
    pscale = parameters['pscale']
    uscale = parameters['uscale']
    ulb = parameters['lb']['u'][:, np.newaxis] 
    uub = parameters['ub']['u'][:, np.newaxis]
    us = parameters['us'] 
    ys = parameters['C'] @ parameters['xs']
    ps = parameters['ps']
    H = parameters['H'] 
    exp_dist_indices = parameters['exp_dist_indices']
    # Start and end for plotting
    (start, end) = plot_range
    q_indices = (1, 3, 5)
    # Get the plant arrays.
    y = np.asarray(plant.y[start:end]).squeeze()*yscale + ys
    z = y @ H.T
    u = np.asarray(plant.u[start:end]).squeeze()*uscale + us
    u[:, q_indices] = u[:, q_indices]/1000
    p = np.asarray(plant.p[start:end]).squeeze()*pscale + ps
    t = np.asarray(plant.t[start:end]).squeeze()/3600
    # Get the controller arrays.
    pest = np.asarray(controller.target_selector.dhats[start:end]).squeeze()
    pest = pest*pscale[exp_dist_indices, np.newaxis].squeeze()
    pest = pest + ps[exp_dist_indices, np.newaxis].squeeze()
    ell = np.asarray(controller.average_stage_costs[start:end]).squeeze()
    # Get the setpoints.
    ysp = setpoints[start:end, :]*yscale + ys
    zsp = ysp @ H.T
    # Upper and lower bounds on the control input.
    ulb = np.repeat(ulb.T, end-start, axis=0)*uscale + us
    ulb[:, q_indices] = ulb[:, q_indices]/1000
    uub = np.repeat(uub.T, end-start, axis=0)*uscale + us
    uub[:, q_indices] = uub[:, q_indices]/1000
    # Return the arrays.
    return (t, z, y, u, pest, p, ulb, uub, zsp, ell)

def _cstrs_cl_comparision_plots(scenarios, 
                                mpc_plants, mpc_controllers,
                                fast_plants, fast_controllers, 
                                fast_controller_name,
                                parameters,
                                plot_range):
    """Data to plot.
    z: 4 outputs with setpoints. y:90 outputs. u:32 inputs.
    """
    # the x coordinate of the ylabels.
    ylabel_xcoordinate = -0.27
    figures = []
    for (scenario, mpc_plant, 
         mpc_controller, fast_plant,
         fast_controller) in zip(scenarios, mpc_plants, 
                                 mpc_controllers, fast_plants,
                                 fast_controllers):
        (setpoints, _) = scenario
        # Get the arrays.
        (t, z_mpc, y_mpc, u_mpc, pest_mpc, 
        p, ulb, uub, zsp, ell_mpc) = _get_cstr_data_for_plotting(mpc_plant, 
                                                  mpc_controller,
                                                  setpoints, 
                                                  parameters, 
                                                  plot_range)
        (t, z_fast, y_fast, u_fast, pest_fast, 
        p, _, _, _, ell_fast) = _get_cstr_data_for_plotting(fast_plant, 
                                                  fast_controller,
                                                  setpoints, 
                                                  parameters, 
                                                  plot_range)
        # Make the Plots.
        figures += plot_inputs(t, (u_mpc, u_fast), ulabels, 
                              ylabel_xcoordinate, ulb, uub, fast_controller_name)
        figures += plot_controlled_outputs(t, (z_mpc, z_fast), 
                              zsp, ylabels, ylabel_xcoordinate,
                              fast_controller_name)
        figures += plot_outputs(t, (y_mpc, y_fast), 
                                ylabels, ylabel_xcoordinate,
                                fast_controller_name)
        figures += plot_disturbances(t, (pest_mpc, pest_fast), p, 
                                     parameters['exp_dist_indices'], 
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
    ylabel_xcoordinate = -0.2
    (u_mpc, u_fast) = u
    input_index = 0
    num_rows = 3
    num_cols = 2
    (figure, axes) = plt.subplots(nrows=num_rows, ncols=num_cols,
                                 sharex='col', figsize=figure_size_cl, 
                                 gridspec_kw = dict(wspace=0.5))
    for (row, column) in itertools.product(range(num_rows), range(num_cols)):
        axes[row, column].plot(t, u_mpc[:, input_index], 'b')
        axes[row, column].plot(t, u_fast[:, input_index], 'g')
        axes[row, column].plot(t, ulb[:, input_index], 'k')
        axes[row, column].plot(t, uub[:, input_index], 'k')
        axes[row, column].set_xlim([np.min(t), np.max(t)])
        axes[row, column].set_ylabel(ulabels[input_index])
        axes[row, column].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        input_index += 1
    # x-axis label.
    axes[2, 0].set_xlabel("Time (hr)")
    axes[2, 1].set_xlabel("Time (hr)")
    figure.legend(labels = ('MPC', fast_controller_name), 
                  loc = (0.30, 0.9), ncol=2)
    # Return the figure,
    return [figure]

def plot_controlled_outputs(t, z, zsp, 
                            ylabels, ylabel_xcoordinate,
                            fast_controller_name):
    """Return 10 figure objects which contain the plots of the inputs."""
    (z_mpc, z_fast) = z
    output_index = 0
    num_rows = 3
    num_cols = 2
    (figure, axes) = plt.subplots(nrows=num_rows, ncols=num_cols,
                                  sharex='col', figsize=figure_size_cl,
                                  gridspec_kw = dict(wspace=0.5))
    for (row, column) in itertools.product(range(num_rows), range(num_cols)):
        axes[row, column].plot(t, z_mpc[:, output_index], 'b')
        axes[row, column].plot(t, z_fast[:, output_index], 'g')
        axes[row, column].plot(t, zsp[:, output_index], 'r--')
        axes[row, column].set_xlim([np.min(t), np.max(t)])
        axes[row, column].set_ylabel(zlabels[output_index], rotation=False)
        axes[row, column].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        output_index += 1
    axes[2, 0].set_xlabel("Time (hr)")
    axes[2, 1].set_xlabel("Time (hr)")
    figure.legend(labels = ('MPC', fast_controller_name), 
                  loc = (0.30, 0.9), ncol=2)
    # Return
    return [figure]

def plot_outputs(t, y, ylabels, 
                 ylabel_xcoordinate, fast_controller_name):
    """Return 1 figure objects which contain the plots of the inputs."""
    (y_mpc, y_fast) = y
    state_index = 0
    num_rows = 6
    num_cols = 2
    (figure, axes) = plt.subplots(nrows=num_rows, ncols=num_cols,
                                  sharex='col', figsize=figure_size_cl,
                                  gridspec_kw = dict(wspace=0.5))
    for (row, column) in itertools.product(range(num_rows), range(num_cols)):
        axes[row, column].plot(t, y_mpc[:, state_index], 'b')
        axes[row, column].plot(t, y_fast[:, state_index], 'g')
        axes[row, column].set_ylabel(ylabels[state_index], rotation=False)
        axes[row, column].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        state_index += 1
    # Get the x-axis label and figure title.
    axes[5, 0].set_xlabel("Time (hr)")
    axes[5, 1].set_xlabel("Time (hr)")
    figure.legend(labels = ('MPC', fast_controller_name), 
                  loc = (0.3, 0.9), ncol=2)
    # Return.
    return [figure]

def plot_disturbances(t, pest, p, exp_dist_indices, 
                      plabels, ylabel_xcoordinate,
                      fast_controller_name):
    """Return a figure object which contain the plots of the inputs."""
    (pest_mpc, pest_fast) = pest
    exp_dist_counter = 0
    num_disturbances = p.shape[1]
    (figure, axes) = plt.subplots(nrows=num_disturbances, ncols=1,
                               sharex=True, figsize=figure_size_cl)
    for disturbance_index in range(num_disturbances):
        axes[disturbance_index].plot(t, p[:, disturbance_index], 'k')
        if disturbance_index in exp_dist_indices:
            axes[disturbance_index].plot(t, pest_mpc[:, exp_dist_counter], 'b')
            axes[disturbance_index].plot(t, pest_fast[:, exp_dist_counter], 'g')
            exp_dist_counter +=1
        axes[disturbance_index].set_ylabel(
            plabels[disturbance_index])
    figure.legend(labels = ('$p$', '$\hat{p}^{MPC}$', 
             '$\hat{p}^{' + fast_controller_name + '}$'), 
                  loc = (0.45, 0.9))
    return [figure]

def plot_average_stage_costs(t, ell, 
                            ylabel_xcoordinate,
                            fast_controller_name):
    """Return a figure object which contain the plots of the inputs."""
    (ell_mpc, ell_fast) = ell
    (figure, axes) = plt.subplots(nrows=1, ncols=1,
                               sharex=True, figsize=figure_size_metrics)
    axes.plot(t, ell_mpc, 'b')
    axes.plot(t, ell_fast, 'g')
    axes.set_ylabel('$\ell_k$')
    figure.legend(labels = ('MPC', fast_controller_name), loc = (0.45, 0.9))
    return [figure]

def print_unstd_pars():

    dim = [36, 224, 224, 224, 6]
    (Nin, Nh, Nnh, Nout) = (dim[0], len(dim)-2, dim[1], dim[-1])
    num_pars = Nin*Nnh + Nnh  + (Nnh**2)*(Nh-1) + Nnh*(Nh-1) + Nnh*Nout + Nout
    print(""" Number of parameters in the 
              unstructued architecture are: """ + str(num_pars))
    return 

def main():
    """ Run these commands when this script is called."""

    # Print Num pars in the unstructured Architecture.
    print_unstd_pars()

    # Get the parameters and the plant object which has the closed-loop data
    # for the two controllers.
    (cstrs_parameters, cstrs_mpc, cstrs_us, 
    cstrs_satdlqr, cstrs_short_horizon, 
    cstrs_train, 
    cstrs_neural_network) = _load_data_for_plots(plant_name='cstrs')

    # Optimal/NN plants/controllers. 
    mpc_plants = cstrs_mpc['plants']
    mpc_controllers = cstrs_mpc['controllers']
    us_controllers = cstrs_us['controllers']
    satdlqr_controllers = cstrs_satdlqr['controllers']
    short_horizon_controllers = cstrs_short_horizon['controllers']
    nn_plants = cstrs_neural_network['plants']
    nn_controllers = cstrs_neural_network['controllers']
    performance_loss = cstrs_neural_network['performance_loss']

    # Number of architectures, NNs, scenarios.
    num_architectures = len(cstrs_train['regulator_dims'])
    num_nns_per_architecture = len(cstrs_train['num_samples'])
    num_scenarios = len(cstrs_parameters['online_test_scenarios'])

    # Create lists to store the figures and the NN metrics.
    nn_labels = ['NN-3-448', 'NN-3-480', 'NN-3-512', 'NN-3-544']
    cl_figures = []
    scenarios = cstrs_parameters['online_test_scenarios']
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
        cl_figures += _cstrs_cl_comparision_plots(scenarios, 
                                 mpc_plants, mpc_controllers,
                                 fast_plants, fast_controllers, 
                                 nn_labels[arch],
                                 cstrs_parameters['cstrs_plant_parameters'],
                                 (0, 720))

    # Make the plots for the closed loop metrics of NNs with 
    # increasing data.
    performance_loss = cstrs_neural_network['performance_loss']
    num_samples = cstrs_train['num_samples']
    cl_metric_figures = plot_nn_vs_num_samples_performance(performance_loss,
                                             num_samples, 
                                             nn_labels,
                                             figure_size_metrics,
                                             right_frac=0.9,
                                             legend_title_location=(0.6, 0.5),
                                             ylabel_xcoordinate=-0.1,
                                             yaxis_log=False)

    # Plot the average stage cost in time.
    time = np.asarray(cstrs_mpc['plants'][0].t[0:-1]).squeeze()/3600
    nn_controllers = PickleTool.load(filename='cstrs_neural_network.pickle', 
                                     type='read')['controllers']
    cl_ct_figures = plot_cl_performance_and_comp_times(time,
                         mpc_controllers,
                         us_controllers, satdlqr_controllers,
                         short_horizon_controllers, 
                         nn_plants, nn_controllers, performance_loss,
                         num_architectures, num_nns_per_architecture,
                         cl_xlabel = 'Time (hr) ', 
                         cl_legend_labels = ['MPC', 'SS', 'satK'] + nn_labels,
                         cl_right_frac = 0.9,
                         ct_right_frac = 0.9,
                         ct_legend_labels = ['MPC'] + nn_labels,
                         figure_size=figure_size_metrics,
                         fig_cl_title_location = (0.35, 0.15),
                         fig_ct_title_location = 'center',
                         cl_us=True,
                         cl_short_horizon=False,
                         cl_yaxis_log=False,
                         ylabel_xcoordinate=-0.1,
                         num_bins=1000)
    # Stuff all the figures into one.
    figures = cl_figures + cl_metric_figures + cl_ct_figures
    with PdfPages('cstrs_comparision_plots.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute the main function
main()