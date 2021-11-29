"""
Module for controller evaluation on different plant models
with different controllers. 
Author: Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import time
import sys
import cvxopt as cvx
import numpy as np 
import scipy
import itertools
from linearMPC import (DenseQPRegulator, online_simulation,
                       LinearMPCController, dlqr)
import matplotlib.pyplot as plt
from python_utils import (figure_size_a4, PickleTool,
                          H5pyTool)
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({'figure.max_open_warning': 0})

def _sample_repeats(num_change, num_simulation_steps,
                       mean_change, sigma_change):
    """ Sample the number of times a repeat in each
    of the sampled vector is required."""
    repeat = sigma_change*np.random.randn(num_change-1) + mean_change
    repeat = np.floor(repeat)
    repeat = np.where(repeat<=0., 0., repeat)
    repeat = np.append(repeat, num_simulation_steps-np.int(np.sum(repeat)))
    return repeat.astype(int)

def sample_prbs_like(*, num_change, num_steps, 
                    lb, ub, mean_change, sigma_change, seed=1):
    """Sample a PRBS like sequence.
    num_change: Number of changes in the signal.
    num_simulation_steps: Number of steps in the signal.
    mean_change: mean_value after which a 
                 change in the signal is desired.
    sigma_change: standard deviation of changes in the signal.
    """
    signal_dimension = lb.shape[0]
    lb = lb.squeeze() # Squeeze the vectors.
    ub = ub.squeeze() # Squeeze the vectors.
    np.random.seed(seed)
    values = (ub-lb)*np.random.rand(num_change, signal_dimension) + lb
    repeat = _sample_repeats(num_change, num_steps,
                             mean_change, sigma_change)
    return np.repeat(values, repeat, axis=0)

def _get_best_nn_plant_controllers(nn_plants, nn_controllers, 
                             performance_loss,
                             num_architectures,
                             num_nns_per_arch, 
                             num_scenarios):
    """ Return the NN controllers for each architecture
        that use larget amount of data."""
    best_nn_plants = []
    best_nn_controllers = []
    best_nn_indices = np.argmin(performance_loss, axis=1)
    for (arch, scenario) in itertools.product(range(num_architectures), 
                                range(num_scenarios)):
        nn_plant = nn_plants[arch*num_nns_per_arch*num_scenarios + 
                    best_nn_indices[arch, scenario]*num_scenarios + scenario]
        nn_controller = nn_controllers[arch*num_nns_per_arch*num_scenarios + 
                    best_nn_indices[arch, scenario]*num_scenarios + scenario]
        best_nn_plants.append(nn_plant)
        best_nn_controllers.append(nn_controller)
    return (best_nn_plants, best_nn_controllers)

def _get_ells_comp_times_for_plotting(Nplot, mpc_controllers,
                                us_controllers, satdlqr_controllers,
                                short_horizon_controllers, 
                                nn_plants, nn_controllers, performance_loss,
                                num_architectures, num_nns_per_architecture):
    """ Get the average stage costs for plotting."""
    # Get the NN controllers which use the most data for training.
    num_scenarios = len(mpc_controllers)
    (_, nn_controllers) = _get_best_nn_plant_controllers(nn_plants, 
                                              nn_controllers,
                                              performance_loss,
                                              num_architectures, 
                                              num_nns_per_architecture,
                                              num_scenarios)

    # Create the matrices in which to store the data to plot.
    ells = np.zeros((Nplot, 4+num_architectures, num_scenarios))
    comp_times = np.zeros((Nplot, 4+num_architectures, num_scenarios))

    for scenario in range(num_scenarios):

        # Get the controllers.
        mpc_controller = mpc_controllers[scenario]
        us_controller = us_controllers[scenario]
        satdlqr_controller = satdlqr_controllers[scenario]
        short_horizon_controller = short_horizon_controllers[scenario]

        # Get the computation times.
        ell_mpc = mpc_controller.average_stage_costs[0:Nplot] 
        ell_mpc = np.asarray(ell_mpc).squeeze()
        ell_us = us_controller.average_stage_costs[0:Nplot] 
        ell_us = np.asarray(ell_us).squeeze()
        ell_satdlqr = satdlqr_controller.average_stage_costs[0:Nplot] 
        ell_satdlqr = np.asarray(ell_satdlqr).squeeze()
        ell_sh = short_horizon_controller.average_stage_costs[0:Nplot] 
        ell_sh = np.asarray(ell_sh).squeeze()

        # Get the stage cost values.
        comp_time_mpc = mpc_controller.computation_times[0:Nplot] 
        comp_time_mpc = np.asarray(comp_time_mpc).squeeze()
        comp_time_us = us_controller.computation_times[0:Nplot] 
        comp_time_us = np.asarray(comp_time_us).squeeze()
        comp_time_satdlqr = satdlqr_controller.computation_times[0:Nplot] 
        comp_time_satdlqr = np.asarray(comp_time_satdlqr).squeeze()
        comp_time_sh= short_horizon_controller.computation_times[0:Nplot] 
        comp_time_sh = np.asarray(comp_time_sh).squeeze()

        # Save the values in the array.
        ells[:, 0, scenario] = ell_mpc
        ells[:, 1, scenario] = ell_us
        ells[:, 2, scenario] = ell_satdlqr
        ells[:, 3, scenario] = ell_sh
        comp_times[:, 0, scenario] = comp_time_mpc
        comp_times[:, 1, scenario] = comp_time_us
        comp_times[:, 2, scenario] = comp_time_satdlqr
        comp_times[:, 3, scenario] = comp_time_sh

        counter = 4
        for arch in range(num_architectures):
            nn_controller = nn_controllers[arch*num_scenarios + scenario]
            ell_nn = nn_controller.average_stage_costs[0:Nplot] 
            ell_nn = np.asarray(ell_nn).squeeze()
            comp_time_nn = nn_controller.computation_times[0:Nplot] 
            comp_time_nn = np.asarray(comp_time_nn).squeeze()
            ells[:, counter, scenario] = ell_nn
            comp_times[:, counter, scenario] = comp_time_nn
            counter +=1

    # Return the arrays.
    return (ells, comp_times)

def plot_cl_performance_and_comp_times(time,
                         mpc_controllers,
                         us_controllers, satdlqr_controllers,
                         short_horizon_controllers, 
                         nn_plants, nn_controllers, performance_loss,
                         num_architectures, num_nns_per_architecture, 
                         *, cl_xlabel, cl_legend_labels,
                         ct_legend_labels,
                         figure_size, fig_cl_title_location,
                         fig_ct_title_location, num_bins,
                         cl_right_frac,
                         ct_right_frac,
                         cl_us=True,
                         cl_short_horizon=True,
                         cl_yaxis_log=False,
                         ylabel_xcoordinate=-0.05):
    """ 
       Script to plot multiple closed-loop metrics
       in one script.
    """
    (ells, comp_times) = _get_ells_comp_times_for_plotting(len(time), 
                                mpc_controllers,
                                us_controllers, satdlqr_controllers,
                                short_horizon_controllers, 
                                nn_plants, nn_controllers, performance_loss,
                                num_architectures, num_nns_per_architecture)
    if not cl_us and not cl_short_horizon: 
        ells = np.delete(ells, (1, 3), axis=1)
        comp_times = np.delete(comp_times, (1, 3), axis=1)
        num_simple_controllers = 1
    elif not cl_us and cl_short_horizon: 
        ells = np.delete(ells, 1, axis=1)
        comp_times = np.delete(comp_times, 1, axis=1)
        num_simple_controllers = 2
    elif cl_us and not cl_short_horizon: 
        ells = np.delete(ells, 3, axis=1)
        comp_times = np.delete(comp_times, 3, axis=1)
        num_simple_controllers = 2
    elif cl_us and cl_short_horizon:
        num_simple_controllers = 3
    num_scenarios = len(mpc_controllers)
    figures = []
    for scenario in range(num_scenarios):
        (figure_cl, axes_cl) = plt.subplots(figsize=figure_size, 
                                        gridspec_kw=dict(right=cl_right_frac))
        (figure_ct, axes_ct) = plt.subplots(figsize=figure_size,
                                        gridspec_kw=dict(right=ct_right_frac))
        for controller in range(1+num_simple_controllers+num_architectures):
            if cl_yaxis_log:
                axes_cl.semilogy(time, ells[:, controller, scenario])
            else:
                axes_cl.plot(time, ells[:, controller, scenario])
            if controller == 0 or controller>num_simple_controllers: 
               axes_ct.hist(comp_times[:, controller, scenario], 
                            bins=num_bins) 
        # Set some plotting parameters for closed-loop performances.
        axes_cl.set_xlabel(cl_xlabel)
        axes_cl.set_ylabel(r'$\Lambda_k$', rotation=False)
        axes_cl.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        axes_cl.set_xlim([np.min(time), np.max(time)])
        figure_cl.legend(labels=cl_legend_labels, 
                         loc=fig_cl_title_location, 
                         ncol=2)
        # Set some plotting parameters for the histogram.
        axes_ct.set_xscale('log')
        axes_ct.set_yscale('linear')
        axes_ct.set_xlabel('Computation times (sec)')
        axes_ct.set_ylabel('Frequency')
        axes_ct.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        figure_ct.legend(labels=ct_legend_labels, 
                         loc=fig_ct_title_location, 
                         ncol=1)
        # Get the figures in a list.
        figures += [figure_cl, figure_ct]
    # Return the figure.
    return figures

def plot_nn_vs_num_samples_performance(performance_loss,
                                    num_samples, legend_labels,
                                    figure_size,
                                    right_frac=None,
                                    legend_title_location=None,
                                    ylabel_xcoordinate=-0.05,
                                    yaxis_log=False):
    """ 
       Script to plot multiple closed-loop metrics
       in one script.
    """
    (num_architectures, _, 
        num_scenarios) = performance_loss.shape
    figures = []
    num_samples = np.asarray(num_samples).squeeze()
    for scenario in range(num_scenarios):
        (figure, axes) = plt.subplots(figsize=figure_size,
                                    gridspec_kw=dict(right=right_frac))
        for architecture in range(num_architectures):
            if yaxis_log:
                axes.semilogy(num_samples, 
                          performance_loss[architecture, :, scenario]) 
            else:
                axes.plot(num_samples, 
                          performance_loss[architecture, :, scenario]) 
        # Set some plotting parameters.
        figure.legend(labels=legend_labels, 
                         loc=legend_title_location, 
                         ncol=1)
        axes.set_xlabel('Number of samples used for training')
        axes.set_ylabel('\% Performance Loss')
        axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        axes.set_xlim([num_samples[0], num_samples[-1]])
        figures.append(figure)
    # Return the figure.
    return figures

def _get_data_for_training(*, data, num_samples, scale=True):
    """ Get the scaling for the states and return the 
        scaled data for training."""
    data = dict(x=data['x'][0:num_samples, :],
                uprev=data['uprev'][0:num_samples, :],
                xs=data['xs'][0:num_samples, :],
                us=data['us'][0:num_samples, :],
                u=data['u'][0:num_samples, :])
    if scale:
        # Get the scaling for the states.
        xmin = np.min(data['x'], axis=0)
        xmax = np.max(data['x'], axis=0)
        xscale = 0.5*(xmax - xmin)
        data['x'] = data['x']/xscale
        data['xs'] = data['xs']/xscale
        return (data, xscale)
    else:
        return data

def _post_process_data(*, data_filename, 
                          num_data_gen_task,
                          num_process_per_task):
    """ Load all the generated data for a given 
        system and compile into one."""
        # Load and concatenate all data.
    training_data = dict(x=[], uprev=[], xs=[], 
                         us=[], u=[], data_gen_time=[])
    for (task, process) in itertools.product(range(num_data_gen_task), 
                                             range(num_process_per_task)):
        sub_data_filename = str(task)+'-'+str(process)+'-'+data_filename
        process_data = H5pyTool.load_training_data(filename=sub_data_filename)
        for key in process_data.keys():
            training_data[key].append(process_data[key])
    # Now concatenate.
    for key in training_data.keys():
        if key == 'data_gen_time':
            training_data[key] = np.mean(np.asarray(training_data[key]))
        else:
            training_data[key] = np.concatenate(training_data[key], axis=0)
    # Finally save.
    H5pyTool.save_training_data(dictionary=training_data, 
                                filename=data_filename)

def _post_process_trained_data(*, data_filename,
                              num_architectures):
    """ Load all the data for the trained neural 
        network and compile into one. """
    trained_data = dict(trained_regulator_weights=[], 
                        data_generation_times=[], 
                        training_times=[],
                        memory_footprints=[], 
                        regulator_dims=[])
    for trained_nn in range(num_architectures):
        trained_nn_filename = str(trained_nn) + '-' + data_filename
        trained_nn_data = PickleTool.load(filename=trained_nn_filename,
                                        type='read')
        for key in trained_data.keys():
            if key in ['trained_regulator_weights', 'regulator_dims']:
                trained_data[key] += trained_nn_data[key]
            if key in ['data_generation_times', 'training_times', 'memory_footprints']:
                trained_data[key] += [trained_nn_data[key][-1]]
    trained_data['num_samples'] = trained_nn_data['num_samples']
    trained_data['num_architectures'] = num_architectures
    trained_data['xscale'] = trained_nn_data['xscale']
    # Save the training data.
    PickleTool.save(data_object=trained_data,
                    filename=data_filename)

def _simulate_scenarios(*, plant_name, controller_name, Nsim, seed=0):
    """ Simulate the plant with the controllers 
        for the desired number of scenarios."""

    # Load data and do an online simulation using the linear MPC controller.
    plant_filename =  plant_name + '_parameters.pickle'
    data_filename = plant_name + '_' + str(controller_name) + '.pickle'
    stdout_filename = plant_name + '_' + str(controller_name) + '.txt'
    online_test_scenarios = PickleTool.load(filename=plant_filename, 
                                        type='read')['online_test_scenarios']

    # Lists to store simulation plant and controller object.
    simulation_plants = []
    simulation_controllers = []
    counter = 0

    # Also compute and print the loss with the 'MPC' controller if needed.
    if controller_name != 'mpc':
        mpc_controller_filename = str(plant_name)+'_mpc.pickle'
        mpc_controllers = PickleTool.load(filename=mpc_controller_filename, 
                                     type='read')['controllers']
        controller_metrics = np.zeros((len(online_test_scenarios),))
        mpc_metrics = np.zeros((len(online_test_scenarios),))
        average_speedups = np.zeros((len(online_test_scenarios),))
        worst_case_speedups = np.zeros((len(online_test_scenarios),))
    else:
        average_comp_time = np.zeros((len(online_test_scenarios), ))
        worst_case_comp_time = np.zeros((len(online_test_scenarios), ))

    # Loop over the scenarios and start the simulations.
    for scenario in online_test_scenarios:

        # Set numpy seed.
        np.random.seed(seed)

        # Get the setpoint and disturbance signal from the scenario
        # and run the simulation.
        (setpoints, disturbances) = scenario
        plant = PickleTool.load(filename=plant_filename, 
                                type='read')['plant']
        controller = PickleTool.load(filename=plant_filename, 
                                     type='read')[controller_name]
        online_simulation(plant, controller, 
                          setpoints=setpoints, 
                          disturbances=disturbances,
                          Nsim=Nsim, stdout_filename=stdout_filename)

        # Save the plant and the controller used in this simulation. 
        simulation_plants.append(plant)
        simulation_controllers.append(controller)

        if controller_name != 'mpc':
            # Get the performance metrics in one place.
            mpc_controller = mpc_controllers[counter]
            mpc_ell = mpc_controller.average_stage_costs[-1].squeeze()
            mpc_metrics[counter] = mpc_ell
            controller_ell = controller.average_stage_costs[-1].squeeze()
            controller_metrics[counter] = controller_ell
            mpc_comp_times = np.asarray(mpc_controller.computation_times)
            avg_mpc_ctime = np.mean(mpc_comp_times)
            best_mpc_ctime = np.min(mpc_comp_times)
            controller_ctimes = np.asarray(controller.computation_times)
            avg_controller_ctime = np.mean(controller_ctimes)
            wrst_controller_ctime = np.max(controller_ctimes)
            average_speedups[counter] = avg_mpc_ctime/avg_controller_ctime
            worst_case_speedups[counter] = best_mpc_ctime/wrst_controller_ctime
        else:
            comp_times = np.asarray(controller.computation_times)
            average_comp_time[counter] = np.mean(comp_times)
            worst_case_comp_time[counter] = np.max(comp_times)
        counter+=1

    if controller_name != 'mpc':
        performance_loss = 100*(controller_metrics - mpc_metrics)
        performance_loss = performance_loss/mpc_metrics
        data_dict = dict(plants=simulation_plants, 
                    controllers=simulation_controllers,
                    performance_loss=performance_loss,
                    average_speedups=average_speedups,
                    worst_case_speedups=worst_case_speedups)
        print("Performance loss by the " + str(controller_name) 
               + " controller is:" + str(performance_loss))
    else:
        data_dict = dict(plants=simulation_plants, 
                         controllers=simulation_controllers,
                         average_comp_time=average_comp_time,
                         worst_case_comp_time=worst_case_comp_time)
    
    # Save the plant instance for plotting.
    PickleTool.save(data_object=data_dict,
                    filename=data_filename)
    # End the simulation of the scenarios and return.
    return (simulation_plants, simulation_controllers)

def _simulate_neural_networks(*, plant_name, Nsim, nnwithuprev=True, seed=0):
    """ Perform closed-loop simulations with the trained 
        neural networks for the cooresponding plant."""

    # Filenames.
    plant_filename = str(plant_name) + '_parameters.pickle'
    training_data_filename = str(plant_name) + '_train.pickle'
    mpc_filename = str(plant_name) + '_mpc.pickle'
    stdout_filename = str(plant_name) + '_neural_network.txt'
    sim_data_filename = str(plant_name) + '_neural_network.pickle'

    # Load plant and optimal simulated controller data.
    plant_parameters = PickleTool.load(filename=plant_filename, 
                                    type='read')
    online_test_scenarios = plant_parameters['online_test_scenarios']
    mpc_controller = plant_parameters['mpc']
    mpc_simulated_controllers = PickleTool.load(filename=mpc_filename, 
                                    type='read')['controllers']

    # Get the trained regulator weights.
    trained_data = PickleTool.load(filename=training_data_filename, 
                                    type='read')
    num_architectures = trained_data['num_architectures']
    num_nns_per_architecture = len(trained_data['num_samples'])
    num_scenarios = len(online_test_scenarios)
    regulator_weights = trained_data['trained_regulator_weights']
    xscale = trained_data['xscale']

    # Create matrices to store performance metrics.
    nn_metrics = np.zeros((num_architectures, 
                           num_nns_per_architecture, 
                           num_scenarios))
    mpc_metrics = np.zeros((1, num_scenarios))
    
    # Create matrices to examine timing performances.
    average_comp_time = np.zeros((num_architectures, num_scenarios))
    worst_case_comp_time = np.zeros((num_architectures, num_scenarios))
    average_speedups = np.zeros((num_architectures, num_scenarios))
    worst_case_speedups = np.zeros((num_architectures, num_scenarios))

    # Create lists to store the plant and controllers objects for the simulations.
    simulation_plants = []
    simulation_controllers = []
    nn_weight_counter = 0
    for (arch, sample) in itertools.product(range(num_architectures), 
                                            range(num_nns_per_architecture)):

        for scenario in range(num_scenarios):

            # Set the random seed at the start of the simulation.
            np.random.seed(seed)
            
            # Get plant/controller and do simulation.
            (setpoints, disturbances) = online_test_scenarios[scenario]
            plant = PickleTool.load(filename=plant_filename, 
                                    type='read')['plant']
            nn_controller = _get_nn_controller(mpc_controller, 
                                        regulator_weights[nn_weight_counter],
                                        xscale, nnwithuprev)
            online_simulation(plant, nn_controller, 
                              setpoints=setpoints, 
                              disturbances=disturbances,
                              Nsim=Nsim, stdout_filename=stdout_filename)
            
            # Save the plant and the controller used in this simulation. 
            simulation_plants.append(plant)
            simulation_controllers.append(nn_controller)

            # Get the performance loss metrics.
            mpc_simulated_controller = mpc_simulated_controllers[scenario]
            mpc_ell = mpc_simulated_controller.average_stage_costs[-1].squeeze()
            mpc_metrics[0, scenario] = mpc_ell
            nn_ell = nn_controller.average_stage_costs[-1].squeeze()
            nn_metrics[arch, sample, scenario] = nn_ell

            # Get the timing data for the NNs which use most data.
            if sample == num_nns_per_architecture-1:
                nn_comp_times = np.asarray(nn_controller.computation_times)
                mpc_comp_times = mpc_simulated_controller.computation_times
                mpc_comp_times = np.asarray(mpc_comp_times)

                # Absolute comp times.
                average_comp_time[arch, scenario] = np.mean(nn_comp_times)
                worst_case_comp_time[arch, scenario] = np.max(nn_comp_times)

                # Speed-ups relative to MPC.
                avg_speedup = np.mean(mpc_comp_times)/np.mean(nn_comp_times)
                wrst_speedup = np.min(mpc_comp_times)/np.max(nn_comp_times)
                average_speedups[arch, scenario] = avg_speedup
                worst_case_speedups[arch, scenario] = wrst_speedup
        nn_weight_counter +=1

    # Get the performance losses and print to stdout.
    performance_loss = 100*(nn_metrics - mpc_metrics)
    performance_loss = performance_loss/mpc_metrics
    print(""" Performance losses by the neural network 
              controllers are: """ + str(performance_loss))

    # Save the plant object which contains the data to plot.
    PickleTool.save(data_object=dict(plants=simulation_plants, 
                                     controllers=simulation_controllers,
                                     performance_loss=performance_loss,
                                     average_comp_time=average_comp_time,
                                     worst_case_comp_time=worst_case_comp_time,
                                     average_speedups=average_speedups,
                                     worst_case_speedups=worst_case_speedups),
                    filename=sim_data_filename)
    return None

def _simulate_neural_network_unstd(*, plant_name, Nsim, 
                                      nnwithuprev=True, seed=0):
    """ Perform closed-loop simulations with the trained 
        neural networks for the cooresponding plant."""

    # Filenames.
    plant_filename = str(plant_name) + '_parameters.pickle'
    training_data_filename = str(plant_name) + '_train_unstd.pickle'
    mpc_filename = str(plant_name) + '_mpc.pickle'
    stdout_filename = str(plant_name) + '_neural_network_unstd.txt'
    sim_data_filename = str(plant_name) + '_neural_network_unstd.pickle'

    # Load plant and optimal simulated controller data.
    plant_parameters = PickleTool.load(filename=plant_filename, 
                                    type='read')
    online_test_scenarios = plant_parameters['online_test_scenarios']
    mpc_controller = plant_parameters['mpc']
    mpc_simulated_controllers = PickleTool.load(filename=mpc_filename, 
                                    type='read')['controllers']

    # Get the trained regulator weights.
    trained_data = PickleTool.load(filename=training_data_filename, 
                                    type='read')
    num_scenarios = len(online_test_scenarios)
    regulator_weights = trained_data['trained_regulator_weights']
    xscale = trained_data['xscale']

    # Create matrices to store performance metrics.
    nn_metrics = np.zeros((1, num_scenarios))
    mpc_metrics = np.zeros((1, num_scenarios))
    
    # Create matrices to examine timing performances.
    average_comp_time = np.zeros((1, num_scenarios))
    worst_case_comp_time = np.zeros((1, num_scenarios))
    average_speedups = np.zeros((1, num_scenarios))
    worst_case_speedups = np.zeros((1, num_scenarios))

    # Create lists to store the plant and controllers 
    # objects for the simulations.
    simulation_plants = []
    simulation_controllers = []

    for scenario in range(num_scenarios):

        # Set the random seed at the start of the simulation.
        np.random.seed(seed)
            
        # Get plant/controller and do simulation.
        (setpoints, disturbances) = online_test_scenarios[scenario]
        plant = PickleTool.load(filename=plant_filename, 
                                type='read')['plant']
        nn_controller = _get_nn_controller_unstd(mpc_controller, 
                                        regulator_weights,
                                        xscale, nnwithuprev)
        online_simulation(plant, nn_controller, 
                          setpoints=setpoints, 
                          disturbances=disturbances,
                          Nsim=Nsim, stdout_filename=stdout_filename)
            
        # Save the plant and the controller used in this simulation. 
        simulation_plants.append(plant)
        simulation_controllers.append(nn_controller)

        # Get the performance loss metrics.
        mpc_simulated_controller = mpc_simulated_controllers[scenario]
        mpc_ell = mpc_simulated_controller.average_stage_costs[-1].squeeze()
        mpc_metrics[0, scenario] = mpc_ell
        nn_ell = nn_controller.average_stage_costs[-1].squeeze()
        nn_metrics[0, scenario] = nn_ell

        # Get the computation times.
        nn_comp_times = np.asarray(nn_controller.computation_times)
        mpc_comp_times = mpc_simulated_controller.computation_times
        mpc_comp_times = np.asarray(mpc_comp_times)

        # Absolute comp times.
        average_comp_time[0, scenario] = np.mean(nn_comp_times)
        worst_case_comp_time[0, scenario] = np.max(nn_comp_times)

        # Speed-ups relative to MPC.
        avg_speedup = np.mean(mpc_comp_times)/np.mean(nn_comp_times)
        wrst_speedup = np.min(mpc_comp_times)/np.max(nn_comp_times)
        average_speedups[0, scenario] = avg_speedup
        worst_case_speedups[0, scenario] = wrst_speedup

    # Get the performance losses and print to stdout.
    performance_loss = 100*(nn_metrics - mpc_metrics)
    performance_loss = performance_loss/mpc_metrics
    print(""" Performance losses by the neural network 
              controllers are: """ + str(performance_loss))

    # Save the plant object which contains the data to plot.
    PickleTool.save(data_object=dict(plants=simulation_plants, 
                                     controllers=simulation_controllers,
                                     performance_loss=performance_loss,
                                     average_speedups=average_speedups,
                                     worst_case_speedups=worst_case_speedups),
                    filename=sim_data_filename)

    # Return the plant and the controllers.
    return (simulation_plants, simulation_controllers)

def _load_data_for_plots(*, plant_name, neural_network_unstd=False):
    "Load the data sets for comparision plots."
    parameters = PickleTool.load(filename=plant_name+'_parameters.pickle', 
                                    type='read')
    mpc = PickleTool.load(filename=plant_name+'_mpc.pickle', 
                                    type='read')
    us = PickleTool.load(filename=plant_name+'_us.pickle', 
                                    type='read')
    satdlqr = PickleTool.load(filename=plant_name+'_satdlqr.pickle', 
                                    type='read')
    short_horizon = PickleTool.load(filename=plant_name+
                                    '_short_horizon.pickle', 
                                    type='read')
    train = PickleTool.load(filename=plant_name+'_train.pickle', 
                                    type='read')
    neural_network = PickleTool.load(filename=plant_name+
                                    '_neural_network.pickle', 
                                    type='read')
    if neural_network_unstd:
        neural_network_unstd = PickleTool.load(filename=plant_name+
                                    '_neural_network_unstd.pickle', 
                                    type='read')
        return_datum = (parameters, mpc, us, satdlqr, short_horizon,
                        train, neural_network, neural_network_unstd)
    else:
        return_datum = (parameters, mpc, us, satdlqr, short_horizon,
                        train, neural_network)
    return return_datum

def _get_nn_controller(optimal_controller, 
                       regulator_weights,
                       xscale, nnwithuprev):
    """ Return a controller object such that we 
        use u = us as the control law. """ 
    return NeuralNetworkController(A=optimal_controller.A, 
                                   B=optimal_controller.B, 
                                   C=optimal_controller.C,
                                   H=optimal_controller.H,     
                                   Qwx=optimal_controller.Qwx, 
                                   Qwd=optimal_controller.Qwd, 
                                   Rv=optimal_controller.Rv, 
                                   xprior=optimal_controller.xprior, 
                                   dprior=optimal_controller.dprior, 
                                   Rs=optimal_controller.Rs,
                                   Qs=optimal_controller.Qs, 
                                   Bd=optimal_controller.Bd, 
                                   Cd=optimal_controller.Cd, 
                                   usp=optimal_controller.usp, 
                                   uprev=optimal_controller.uprev, 
                                   ulb=optimal_controller.ulb,
                                   uub=optimal_controller.uub, regulator_weights=regulator_weights,
                                   xscale=xscale,
                                   nnwithuprev=nnwithuprev,
                                   Q=optimal_controller.Q, 
                                   R=optimal_controller.R, 
                                   S=optimal_controller.S)

def _get_nn_controller_unstd(optimal_controller, 
                       regulator_weights,
                       xscale, nnwithuprev):
    """ Return a controller object such that we 
        use u = us as the control law. """ 
    return NeuralNetworkControllerUnstd(A=optimal_controller.A, 
                                   B=optimal_controller.B, 
                                   C=optimal_controller.C,
                                   H=optimal_controller.H,     
                                   Qwx=optimal_controller.Qwx, 
                                   Qwd=optimal_controller.Qwd, 
                                   Rv=optimal_controller.Rv, 
                                   xprior=optimal_controller.xprior, 
                                   dprior=optimal_controller.dprior, 
                                   Rs=optimal_controller.Rs,
                                   Qs=optimal_controller.Qs, 
                                   Bd=optimal_controller.Bd, 
                                   Cd=optimal_controller.Cd, 
                                   usp=optimal_controller.usp, 
                                   uprev=optimal_controller.uprev, 
                                   ulb=optimal_controller.ulb,
                                   uub=optimal_controller.uub, regulator_weights=regulator_weights,
                                   xscale=xscale,
                                   nnwithuprev=nnwithuprev,
                                   Q=optimal_controller.Q, 
                                   R=optimal_controller.R, 
                                   S=optimal_controller.S)

def _get_satdlqr_controller(optimal_controller):
    """ Return a controller object such that we 
        use sat(Kx) as the control law. """ 
    return SatDlqrController(A=optimal_controller.A, B=optimal_controller.B, 
                             C=optimal_controller.C, 
                             H=optimal_controller.H,
                             Qwx=optimal_controller.Qwx, 
                             Qwd=optimal_controller.Qwd, 
                             Rv=optimal_controller.Rv, 
                             xprior=optimal_controller.xprior, 
                             dprior=optimal_controller.dprior, Rs=optimal_controller.Rs,
                             Qs=optimal_controller.Qs, 
                             Bd=optimal_controller.Bd, 
                             Cd=optimal_controller.Cd, 
                             usp=optimal_controller.usp, 
                             uprev=optimal_controller.uprev, 
                             ulb=optimal_controller.ulb,
                             uub=optimal_controller.uub,
                             Q=optimal_controller.Q, R=optimal_controller.R, 
                             S=optimal_controller.S)

def _get_short_horizon_controller(optimal_controller, N):
    """ Return a controller object such that we 
        use an MPC controller with a short horizon. """ 
    return LinearMPCController(A=optimal_controller.A, B=optimal_controller.B, 
                               C=optimal_controller.C, 
                               H=optimal_controller.H,
                               Qwx=optimal_controller.Qwx, 
                               Qwd=optimal_controller.Qwd, 
                               Rv=optimal_controller.Rv, 
                               xprior=optimal_controller.xprior, 
                               dprior=optimal_controller.dprior, Rs=optimal_controller.Rs,
                               Qs=optimal_controller.Qs, 
                               Bd=optimal_controller.Bd, 
                               Cd=optimal_controller.Cd, 
                               usp=optimal_controller.usp, 
                               uprev=optimal_controller.uprev, 
                               Q=optimal_controller.Q, R=optimal_controller.R, 
                               S=optimal_controller.S, N=N,
                               ulb=optimal_controller.ulb,
                               uub=optimal_controller.uub)

def _get_us_controller(optimal_controller):
    """ Return a controller object such that we 
        use u = us as the control law. """ 
    return SteadyStateController(A=optimal_controller.A, 
                             B=optimal_controller.B, 
                             C=optimal_controller.C,
                             H=optimal_controller.H, 
                             Qwx=optimal_controller.Qwx, 
                             Qwd=optimal_controller.Qwd, 
                             Rv=optimal_controller.Rv, 
                             xprior=optimal_controller.xprior, 
                             dprior=optimal_controller.dprior, Rs=optimal_controller.Rs,
                             Qs=optimal_controller.Qs, 
                             Bd=optimal_controller.Bd, 
                             Cd=optimal_controller.Cd, 
                             usp=optimal_controller.usp, 
                             uprev=optimal_controller.uprev, 
                             ulb=optimal_controller.ulb,
                             uub=optimal_controller.uub,
                             Q=optimal_controller.Q, R=optimal_controller.R, 
                             S=optimal_controller.S)

def relu(x):
    """Quick custom numpy relu function."""
    return np.where(x<0, 0., x)

class NeuralNetworkController(LinearMPCController):
    """Class which uses the trained neural network weights
    for closed-loop simulations."""
    def __init__(self, *, A, B, C, H,
                 Qwx, Qwd, Rv, xprior, dprior,
                 Rs, Qs, Bd, Cd, usp, uprev,
                 ulb, uub, regulator_weights,
                 xscale, nnwithuprev, Q, R, S):
        
        # Save linear system matrices.
        self.A = A
        self.B = B
        self.C = C
        self.H = H

        # Sizes.
        self.Nx = A.shape[0]
        self.Nu = B.shape[1]
        self.Ny = C.shape[0]
        self.Nd = Bd.shape[1]

        # Kalman filter options.
        self.Qwx = Qwx
        self.Qwd = Qwd
        self.Rv = Rv
        self.xprior = xprior
        self.dprior = dprior

        # Target selector options.
        self.Qs = Qs
        self.Rs = Rs
        self.Bd = Bd
        self.Cd = Cd
        self.usp = usp

        # Regulator options.
        self.uprev = uprev
        self.ulb = ulb
        self.uub = uub
        self.Q = Q
        self.R = R
        self.S = S
        self.regulator_weights = regulator_weights
        self.xscale = xscale[:, np.newaxis]
        self.nnwithuprev = nnwithuprev

        # Instantiate the Kalman Filter and the Target Selector. 
        self.filter = LinearMPCController.setup_filter(A=A, B=B, C=C, Bd=Bd,
                                        Cd=Cd, Qwx=Qwx, Qwd=Qwd, Rv=Rv,
                                        xprior=xprior, dprior=dprior)
        self.target_selector = LinearMPCController.setup_target_selector(A=A, 
                                        B=B, C=C, H=H, Bd=Bd, Cd=Cd, usp=usp, Qs=Qs, Rs=Rs, ulb=ulb, uub=uub)

        # Form the augmented matrices to compute the infinite horizon cost to go.
        aug_mats = LinearMPCController.get_augmented_matrices_for_regulator(A,  
                                                                B, Q, R, S)
        (_, _, self.Qaug, self.Raug, self.Maug) = aug_mats
        # List objects to store data.
        self.computation_times = []
        self.average_stage_costs = [np.zeros((1, 1))]

    def control_law(self, ysp, y):
        """ Takes the measurement and the previous control input
            and compute the current control input.
        """
        (xhat, dhat) = LinearMPCController.get_state_estimates(self.filter, 
                                                               y, self.uprev, self.Nx)
        (xs, us) = LinearMPCController.get_target_pair(self.target_selector, 
                                                       ysp, dhat)
        tstart = time.time()
        (xhat_scaled, xs_scaled) = self._get_scaled_x_xs(xhat, xs)
        useq_nn = self._get_control_input(xhat_scaled, self.uprev,
                                          xs_scaled, us)
        tend = time.time()
        avg_ell = LinearMPCController.get_updated_average_stage_cost(xhat, 
                    self.uprev, xs, us, useq_nn[0:self.Nu, :], 
                    self.Qaug, self.Raug, self.Maug, 
                    self.average_stage_costs[-1], len(self.average_stage_costs))
        self.average_stage_costs.append(avg_ell)
        self.uprev = useq_nn[0:self.Nu, :]
        self.computation_times.append(tend - tstart)
        return self.uprev

    def _get_scaled_x_xs(self, x, xs):
        """ Return the scaled states to feed to the NN
            controller. """
        return (x/self.xscale, xs/self.xscale)

    def _get_control_input(self, x, uprev, xs, us):
        """ Get the control input by forward propagating 
            and squashing the output of the NN."""
        # Get the control input.
        u = self._get_regulator_nn_output(x, uprev, xs, us)
        u = u - self._get_regulator_nn_output(xs, us, xs, us)
        u = self._clip_control_input(us + u)
        return u

    def _get_regulator_nn_output(self, x, uprev, xs, us):
        """Get the output of the regulator Neural Network."""
        if self.nnwithuprev:
            u = np.concatenate((x, uprev, xs, us), axis=0)
        else:
            u = np.concatenate((x, xs, us), axis=0)
        for i in range(0, len(self.regulator_weights)-1, 2):
            (W, b) = self.regulator_weights[i:i+2]
            u = relu(W.T @ u + b[:, np.newaxis])
        return self.regulator_weights[-1].T @ u

    def _clip_control_input(self, u):
        """Perform clipping to the control input."""
        u = np.where(u > self.uub, self.uub, u)
        u = np.where(u < self.ulb, self.ulb, u)
        return u


class NeuralNetworkControllerUnstd(NeuralNetworkController):
    """Class which uses the trained neural network weights
    for closed-loop simulations."""
    def _get_regulator_nn_output(self, x, uprev, xs, us):
        """Get the output of the regulator Neural Network."""
        if self.nnwithuprev:
            u = np.concatenate((x, uprev, xs, us), axis=0)
        else:
            u = np.concatenate((x, xs, us), axis=0)
        for i in range(0, len(self.regulator_weights)-2, 2):
            (W, b) = self.regulator_weights[i:i+2]
            u = relu(W.T @ u + b[:, np.newaxis])
        (Wf, bf) = self.regulator_weights[-2:]
        return Wf.T @ u + bf[:, np.newaxis]

    def _get_control_input(self, x, uprev, xs, us):
        """ Get the control input by forward propagating 
            and squashing the output of the NN."""
        # Get the control input.
        u = self._get_regulator_nn_output(x, uprev, xs, us)
        u = self._clip_control_input(u)
        return u

class SatDlqrController(LinearMPCController):
    """Class which uses the trained neural network weights
    for closed-loop simulations."""
    def __init__(self, *, A, B, C, H, 
                 Qwx, Qwd, Rv, xprior, dprior,
                 Rs, Qs, Bd, Cd, usp, uprev,
                 ulb, uub,
                 Q, R, S):
        
        # Save linear system matrices.
        self.A = A
        self.B = B
        self.C = C
        self.H = H

        # Sizes.
        self.Nx = A.shape[0]
        self.Nu = B.shape[1]
        self.Ny = C.shape[0]
        self.Nd = Bd.shape[1]

        # Kalman filter options.
        self.Qwx = Qwx
        self.Qwd = Qwd
        self.Rv = Rv
        self.xprior = xprior
        self.dprior = dprior

        # Target selector options.
        self.Qs = Qs
        self.Rs = Rs
        self.Bd = Bd
        self.Cd = Cd
        self.usp = usp

        # Regulator options.
        self.uprev = uprev
        self.ulb = ulb
        self.uub = uub
        self.Q = Q
        self.R = R
        self.S = S

        # Instantiate the Kalman Filter and the Target Selector. 
        self.filter = LinearMPCController.setup_filter(A=A, B=B, C=C, 
                                                       Bd=Bd, Cd=Cd,
                                                       Qwx=Qwx, Qwd=Qwd, Rv=Rv,
                                                       xprior=xprior, dprior=dprior)
        self.target_selector = LinearMPCController.setup_target_selector(A=A, 
                                                        B=B, C=C, H=H, 
                                                        Bd=Bd, Cd=Cd, 
                                                        usp=usp, Qs=Qs, Rs=Rs, 
                                                        ulb=ulb, uub=uub)

        # Form the augmented matrices to compute the infinite horizon cost to go.
        aug_mats = LinearMPCController.get_augmented_matrices_for_regulator(A, 
                                                                    B, Q, R, S)
        (Aaug, Baug, self.Qaug, self.Raug, self.Maug) = aug_mats
        (self.Kaug, _) = dlqr(Aaug, Baug, self.Qaug, self.Raug, self.Maug)

        # List objects to store data.
        self.computation_times = []
        self.average_stage_costs = [np.zeros((1, 1))]

    def control_law(self, ysp, y):
        """ Takes the measurement and the previous control input
            and compute the current control input.
        """
        (xhat, dhat) = LinearMPCController.get_state_estimates(self.filter, 
                                                               y, self.uprev, self.Nx)
        (xs, us) = LinearMPCController.get_target_pair(self.target_selector, 
                                                      ysp, dhat)
        tstart = time.time()
        uKx = self.Kaug @ np.concatenate((xhat-xs, self.uprev-us),axis=0) + us
        uKx = self._clip_control_input(uKx)
        tend = time.time()
        avg_ell = LinearMPCController.get_updated_average_stage_cost(xhat, 
                self.uprev, xs, us, uKx, self.Qaug, self.Raug, self.Maug, 
                    self.average_stage_costs[-1], len(self.average_stage_costs))
        self.average_stage_costs.append(avg_ell)
        self.uprev = uKx
        self.computation_times.append(tend-tstart)
        return self.uprev

    def _clip_control_input(self, u):
        """Perform clipping to the control input."""
        u = np.where(u > self.uub, self.uub, u)
        u = np.where(u < self.ulb, self.ulb, u)
        return u

class SteadyStateController(LinearMPCController):
    """Class which uses the trained neural network weights
    for closed-loop simulations."""
    def __init__(self, *, A, B, C, H, 
                 Qwx, Qwd, Rv, xprior, dprior,
                 Rs, Qs, Bd, Cd, usp, uprev,
                 ulb, uub,
                 Q, R, S):
        
        # Save linear system matrices.
        self.A = A
        self.B = B
        self.C = C
        self.H = H

        # Sizes.
        self.Nx = A.shape[0]
        self.Nu = B.shape[1]
        self.Ny = C.shape[0]
        self.Nd = Bd.shape[1]

        # Kalman filter options.
        self.Qwx = Qwx
        self.Qwd = Qwd
        self.Rv = Rv
        self.xprior = xprior
        self.dprior = dprior

        # Target selector options.
        self.Qs = Qs
        self.Rs = Rs
        self.Bd = Bd
        self.Cd = Cd
        self.usp = usp

        # Regulator options.
        self.uprev = uprev
        self.ulb = ulb
        self.uub = uub
        self.Q = Q
        self.R = R
        self.S = S

        # Instantiate the Kalman Filter and the Target Selector. 
        self.filter = LinearMPCController.setup_filter(A=A, B=B, C=C, 
                                                       Bd=Bd, Cd=Cd,
                                                       Qwx=Qwx, Qwd=Qwd, Rv=Rv,
                                                       xprior=xprior, dprior=dprior)
        self.target_selector = LinearMPCController.setup_target_selector(A=A, 
                                                        B=B, C=C, H=H, 
                                                        Bd=Bd, Cd=Cd, 
                                                        usp=usp, Qs=Qs, Rs=Rs, 
                                                        ulb=ulb, uub=uub)

        # Form the augmented matrices to compute the infinite horizon cost to go.
        aug_mats = LinearMPCController.get_augmented_matrices_for_regulator(A, 
                                                                    B, Q, R, S)
        (_, _, self.Qaug, self.Raug, self.Maug) = aug_mats

        # List objects to store data.
        self.computation_times = []
        self.average_stage_costs = [np.zeros((1, 1))]

    def control_law(self, ysp, y):
        """ Takes the measurement and the previous control input
            and compute the current control input.
        """
        (xhat, dhat) = LinearMPCController.get_state_estimates(self.filter, 
                                                               y, self.uprev, self.Nx)
        (xs, us) = LinearMPCController.get_target_pair(self.target_selector, 
                                                      ysp, dhat)
        tstart = time.time()
        u = us
        tend = time.time()
        avg_ell = LinearMPCController.get_updated_average_stage_cost(xhat, 
                self.uprev, xs, us, u, self.Qaug, self.Raug, self.Maug, 
                    self.average_stage_costs[-1], len(self.average_stage_costs))
        self.average_stage_costs.append(avg_ell)
        self.uprev = u
        self.computation_times.append(tend-tstart)
        return self.uprev