# [makes] pickle
""" 
Generate the parameters for the offline data generation and 
the online simulation for the crude distillation unit model. 
"""
#import custompath
#custompath.add()
import sys
sys.path.append('lib/')
import matio
import scipy.linalg
import numpy as np
from python_utils import PickleTool
from controller_evaluation import (sample_prbs_like, 
                                   _get_satdlqr_controller,
                                   _get_us_controller,
                                   _get_short_horizon_controller)
from linearMPC import (LinearPlantSimulator, LinearMPCController,
                       OfflineSimulator)

def _get_scaled_parameters(model, dist_indices, dist_scaling):
    """ Get the scaling. Both u and y are scaled as:
    unew = u/(0.5*(uub-ulb), ynew = y/(0.5*(yub-ylb))
    and B = B*uscale, C = C/yscale.
    """
    # Extract out the things to be scaled.
    A = model['A']
    B = model['B']
    C = model['C']
    lb = dict(u=model['ulb']-model['us'], y=model['ylb']-model['ys'])
    ub = dict(u=model['uub']-model['us'], y=model['yub']-model['ys'])
    
    # Get the scaling numbers.
    uscale = 0.5*(ub['u']-lb['u'])
    yscale = 0.5*(ub['y']-lb['y'])

    # Scale the bounds.
    lb['u'] = lb['u']/uscale
    ub['u'] = ub['u']/uscale
    lb['y'] = lb['y']/yscale
    ub['y'] = ub['y']/yscale
    
    # Scale the matrices.
    B = B @ np.diag(uscale.squeeze())
    C = np.diag(1/yscale.squeeze()) @ C

    # Get the plant disturbance model.
    Bp = np.take(B, dist_indices, axis=1)#*dist_scaling

    # Return the required parameters.
    return (uscale, yscale, A, B, C, Bp, lb, ub)

def _get_cdu_plant(model, dist_indices, dist_scaling):
    """ Function to get the CDU plant and the scaling
        for plotting. """
    # Get the scaled model.
    scaled_parameters = _get_scaled_parameters(model, dist_indices,
                                              dist_scaling)
    (uscale, yscale, A, B, C, Bp, lb, ub) = scaled_parameters

    # The measurement error.
    Rv = (1e-20)*np.eye(C.shape[0])

    # Instantiate the CDU class.
    return (LinearPlantSimulator(A=A, B=B, C=C, Bp=Bp, Rv=Rv,
            sample_time=1., x0=np.zeros((A.shape[0],1))),
            uscale, yscale, lb, ub)

def _get_cdu_mpc_controller(plant, lb, ub):
    """ Get the optimal MPC controller for the crude distillation
    unit model."""

    # Get shapes and sizes.
    (Nx, Nu) = plant.B.shape
    (Ny, _) = plant.C.shape
    Nz = 4
    H = np.zeros((0, Ny))
    Nd = plant.Bp.shape[1]

    # Disturbance model.
    Bd = plant.Bp
    Cd = np.zeros((Ny, Nd))
    
    # Kalman-Filter parameters.
    Qwx = (1e-16)*np.eye(Nx)
    Qwd = (1e-6)*np.eye(Nd)
    xprior = plant.x[-1]
    dprior = np.zeros((Nd, 1))
    uprev = np.zeros((Nu, 1))
    Rv = np.diag(plant.measurement_noise_std.squeeze())**2

    # Target selector parameters.
    Rs = 1e-6*np.eye(Nu)
    Qs = scipy.linalg.block_diag((1e-16)*np.eye(Ny-Nz), np.eye(Nz))
    usp = np.zeros((Nu, 1))

    # Regulator parameters.
    Q = 2*(plant.C.T @ plant.C)
    R = 0.1*np.eye(Nu)
    S = 0*np.eye(Nu)
    N = 10
    
    # The input constraints.
    ulb = lb['u']
    uub = ub['u']

    # Construct a controller and return.
    return LinearMPCController(A=plant.A, B=plant.B, C=plant.C, 
                               H=H, Qwx=Qwx, Qwd=Qwd, Rv=Rv, 
                               xprior=xprior, dprior=dprior,
                               Rs=Rs, Qs=Qs, Bd=Bd, Cd=Cd, usp=usp, uprev=uprev,
                               Q=Q, R=R, S=S, N=N, ulb=ulb, uub=uub)

def _get_cdu_offline_simulator(controller, dist_indices, dist_scaling,
                               lb, ub, Nsim, num_data_gen_task, num_process_per_task,
                               seed=1, conservative_factor=1.05):
    """Get the offline model simulator for the crude distillation unit."""
    # Shapes and sizes.
    (Ny, _) = controller.C.shape
    Nz = 4
    H = np.concatenate((np.zeros((Nz, Ny-Nz)), np.eye(Nz)), axis=1)

    # Bounds for the set-points.
    setpoint_lb = (H @ lb['y'])*conservative_factor
    setpoint_ub = (H @ ub['y'])*conservative_factor

    # Bounds for the disturbances.
    disturbance_lb = np.take(lb['u'], dist_indices)
    disturbance_lb = disturbance_lb*dist_scaling*conservative_factor
    disturbance_ub = np.take(ub['u'], dist_indices)
    disturbance_ub = disturbance_ub*dist_scaling*conservative_factor

    # Generate the setpoints.
    setpoints = sample_prbs_like(num_change=750, num_steps=Nsim, 
                                 lb=setpoint_lb, ub=setpoint_ub,
                                 mean_change=400, sigma_change=1, seed=seed)
    setpoints = np.concatenate((np.zeros((Nsim, Ny-Nz)), setpoints), axis=1)

    # Generate the disturbances. 
    disturbances = sample_prbs_like(num_change=1500, num_steps=Nsim, 
                                lb=disturbance_lb, ub=disturbance_ub,
                                mean_change=200, sigma_change=1, seed=seed+1)

    # Construct an offline simulator and return. 
    return  OfflineSimulator(A=controller.A, B=controller.B, C=controller.C,
                             H=controller.H, Rs=controller.Rs, Qs=controller.Qs,
                             Bd=controller.Bd,
                             Cd=controller.Cd, usp=controller.usp, uprev=controller.usp,
                             Q=controller.Q, R=controller.R, S=controller.S,
                             ulb=controller.ulb, uub=controller.uub, N=controller.N, 
                             xprior=controller.xprior, setpoints=setpoints, 
                             disturbances=disturbances,
                             num_data_gen_task=num_data_gen_task,
                             num_process_per_task=num_process_per_task)

def _get_cdu_online_scenarios(controller, dist_indices,
                              dist_scaling, lb, ub, Nsim, 
                              tsteps_steady=10, seed=10):
    """Generate the set-points and the disturbances for on-line testing."""

    # Shapes and sizes.
    (Ny, _) = controller.C.shape
    Nz = 4
    Np = len(dist_indices)
    H = np.concatenate((np.zeros((Nz, Ny-Nz)), np.eye(Nz)), axis=1)

    # Bounds for the set-points.
    setpoint_lb = (H @ lb['y'])
    setpoint_ub = (H @ ub['y'])

    # Bounds for the disturbances.
    disturbance_lb = np.take(lb['u'], dist_indices)*dist_scaling
    disturbance_ub = np.take(ub['u'], dist_indices)*dist_scaling

    # Generate the setpoints.
    setpoints = sample_prbs_like(num_change=24, num_steps=Nsim, 
                                 lb=setpoint_lb, ub=setpoint_ub,
                                 mean_change=120, sigma_change=2, seed=seed)
    setpoints = np.concatenate((np.zeros((Nsim, Ny-Nz)), setpoints), axis=1)
    setpoints[0:tsteps_steady, :] = np.zeros((tsteps_steady, Ny))

    # Generate the disturbances. 
    disturbances = sample_prbs_like(num_change=48, num_steps=Nsim, 
                                    lb=disturbance_lb, ub=disturbance_ub,
                                    mean_change=60, sigma_change=1, 
                                    seed=seed+1)
    disturbances[0:tsteps_steady, :] = np.zeros((tsteps_steady, Np))

    # Compile into a list of scenarios.
    scenarios = [(setpoints.copy(), disturbances)]

    # Return the set-points and the disturbances. 
    return scenarios

if __name__ == "__main__":
    # Load the unscaled model, and the input and the output constraints.
    dist_indices = (0, 6, 23, 30, 31)
    dist_scaling = np.array([[5., 20., 20., 20., 20.]])
    model = matio.loadmat('CDU_Model.mat', squeeze=False)
    # Get the plant.
    (cdu, uscale, 
     yscale, lb, ub) = _get_cdu_plant(model, dist_indices, dist_scaling)
    cdu_mpc_controller = _get_cdu_mpc_controller(cdu, lb, ub)
    cdu_satdlqr_controller = _get_satdlqr_controller(cdu_mpc_controller)
    cdu_us_controller = _get_us_controller(cdu_mpc_controller)
    cdu_sh_controller = _get_short_horizon_controller(cdu_mpc_controller, N=3)
    cdu_offline_simulator = _get_cdu_offline_simulator(cdu_mpc_controller,
                                                       dist_indices,
                                                       dist_scaling,
                                                       lb, ub, Nsim=300000, 
                                            num_data_gen_task=int(sys.argv[1]),
                                            num_process_per_task=1)
    cdu_scenarios = _get_cdu_online_scenarios(cdu_mpc_controller, 
                                              dist_indices,
                                              dist_scaling,
                                              lb, ub, 
                                              Nsim=2880, seed=12)
    cdu_plant_parameters = dict(uscale=uscale, yscale=yscale, lb=lb, ub=ub,
                                dist_indices=dist_indices, 
                                us=model['us'], ys=model['ys'])
    # Create a dictionary with the required data.
    cdu_parameters = dict(plant=cdu, mpc=cdu_mpc_controller,
                          us=cdu_us_controller,
                          satdlqr=cdu_satdlqr_controller,
                          short_horizon=cdu_sh_controller,
                          offline_simulator=cdu_offline_simulator,
                          online_test_scenarios=cdu_scenarios,
                          cdu_plant_parameters=cdu_plant_parameters)
    # Save data.
    PickleTool.save(data_object=cdu_parameters, 
                    filename='cdu_parameters.pickle')