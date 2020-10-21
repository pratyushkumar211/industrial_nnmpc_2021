# [depends] %LIB%/linearMPC.py
# [depends] %LIB%/nonlinearMPC.py
# [depends] %LIB%/controller_evaluation.py
# [depends] %LIB%/python_utils.py
# [makes] pickle
""" 
Generate necessary parameters for the offline data generation and 
the online simulation of the CSTRs in series with a Flash model.
"""
import sys
sys.path.append('lib/')
import mpctools as mpc
import numpy as np
from python_utils import PickleTool
from controller_evaluation import (sample_prbs_like,
                                   _get_us_controller,
                                   _get_satdlqr_controller,
                                   _get_short_horizon_controller)
from linearMPC import (LinearMPCController, 
                       OfflineSimulator,
                       LinearPlantSimulator)
from nonlinearMPC import NonlinearPlantSimulator

def _cstrs_ode(x, u, p, parameters):

    # Extract the parameters.
    alphaA = parameters['alphaA']
    alphaB = parameters['alphaB']
    alphaC = parameters['alphaC']
    pho = parameters['pho']
    Cp = parameters['Cp']
    Ar = parameters['Ar']
    Am = parameters['Am']
    Ab = parameters['Ab']
    kr = parameters['kr']
    km = parameters['km']
    kb = parameters['kb']
    delH1 = parameters['delH1']
    delH2 = parameters['delH2']
    EbyR = parameters['EbyR']
    k1star = parameters['k1star']
    k2star = parameters['k2star']
    Td = parameters['Td']
    uscale = parameters['uscale']
    pscale = parameters['pscale']
    xs = parameters['xs']
    us = parameters['us']
    ps = parameters['ps']

    # Extract the plant states into meaningful names.
    x = x + xs
    (Hr, xAr, xBr, Tr) = x[0:4]
    (Hm, xAm, xBm, Tm) = x[4:8]
    (Hb, xAb, xBb, Tb) = x[8:12]
    (F0, Qr, F1, Qm, D, Qb) = (u*uscale + us)[0:6]
    (xA0, xB0, xA1, xB1, T0) = (p*pscale + ps)[0:5]

    # The flash vapor phase mass fractions.
    denominator = alphaA*xAb + alphaB*xBb + alphaC*(1 - xAb - xBb) 
    xAd = alphaA*xAb/denominator
    xBd = alphaB*xBb/denominator

    # The outlet mass flow rates.
    Fr = kr*np.sqrt(Hr)
    Fm = km*np.sqrt(Hm)
    Fb = kb*np.sqrt(Hb)
    Fp = 0.01*D

    # The rate constants.
    k1r = k1star*np.exp(-EbyR/Tr)
    k2r = k2star*np.exp(-EbyR/Tr)
    k1m = k1star*np.exp(-EbyR/Tm)
    k2m = k2star*np.exp(-EbyR/Tm)

    # Write get the 12 odes.
    # CSTR-1.
    dHrbydt = (F0 + D - Fr)/(pho*Ar)
    dxArbydt = (F0*(xA0 - xAr) + D*(xAd - xAr))/(pho*Ar*Hr) - k1r*xAr
    dxBrbydt = (F0*(xB0 - xBr) + D*(xBd - xBr))/(pho*Ar*Hr) + k1r*xAr - k2r*xBr
    dTrbydt = (F0*(T0 - Tr) + D*(Td - Tr))/(pho*Ar*Hr) 
    dTrbydt = dTrbydt - (1/Cp)*(k1r*xAr*delH1 + k2r*xBr*delH2) 
    dTrbydt = dTrbydt + Qr/(pho*Ar*Cp*Hr)

    # CSTR-2.
    dHmbydt = (Fr + F1 - Fm)/(pho*Am)
    dxAmbydt = (Fr*(xAr - xAm) + F1*(xA1 - xAm))/(pho*Am*Hm) - k1m*xAm
    dxBmbydt = (Fr*(xBr - xBm) + F1*(xB1 - xBm))/(pho*Am*Hm) + k1m*xAm - k2m*xBm
    dTmbydt = (Fr*(Tr - Tm) + F1*(T0 - Tm))/(pho*Am*Hm) 
    dTmbydt = dTmbydt - (1/Cp)*(k1m*xAm*delH1 + k2m*xBm*delH2) 
    dTmbydt = dTmbydt + Qm/(pho*Am*Cp*Hm)

    # Flash.
    dHbbydt = (Fm - Fb - D - Fp)/(pho*Ab)
    dxAbbydt = (Fm*(xAm - xAb) - (D + Fp)*(xAd - xAb))/(pho*Ab*Hb)
    dxBbbydt = (Fm*(xBm - xBb) - (D + Fp)*(xBd - xBb))/(pho*Ab*Hb)
    dTbbydt = (Fm*(Tm - Tb))/(pho*Ab*Hb) + Qb/(pho*Ab*Cp*Hb)

    # Return the derivative.
    return np.array([dHrbydt, dxArbydt, dxBrbydt, dTrbydt, 
                     dHmbydt, dxAmbydt, dxBmbydt, dTmbydt,
                     dHbbydt, dxAbbydt, dxBbbydt, dTbbydt])

def _cstrs_measurement(x, parameters):
    """ The measurement function."""
    yscale = parameters['yscale']
    C = np.diag(1/yscale.squeeze()) @ parameters['C']
    # Return the measurements.
    return C.dot(x)

def _get_cstrs_parameters():
    """ Get the parameter values for the 
        CSTRs in a series with a Flash example.
        The sample time is in seconds."""

    sample_time = 10. 
    Nx = 12
    Nu = 6
    Np = 5 
    Ny = 12
    
    # Parameters.
    parameters = {}
    parameters['alphaA'] = 3.5 
    parameters['alphaB'] = 1.1
    parameters['alphaC'] = 0.5
    parameters['pho'] = 50. # Kg/m^3 # edited.
    parameters['Cp'] = 3. # KJ/(Kg-K) # edited.
    parameters['Ar'] = 0.3 # m^2 
    parameters['Am'] = 2. # m^2 
    parameters['Ab'] = 4. # m^2 
    parameters['kr'] = 2.5 # m^2
    parameters['km'] = 2.5 # m^2
    parameters['kb'] = 1.5 # m^2
    parameters['delH1'] = -40 # kJ/Kg 
    parameters['delH2'] = -50 # kJ/Kg
    parameters['EbyR'] = 150 # K 
    parameters['k1star'] = 4e-4 # 1/sec #edited.
    parameters['k2star'] = 1.8e-6 # 1/sec #edited.
    parameters['Td'] = 313 # K # edited.

    # Store the dimensions.
    parameters['Nx'] = Nx
    parameters['Nu'] = Nu
    parameters['Ny'] = Ny
    parameters['Np'] = Np

    # Sample Time.
    parameters['sample_time'] = sample_time

    # Get the steady states.
    parameters['xs'] = np.array([178.56, 1, 0, 313, 
                                 190.07, 1, 0, 313, 
                                 5.17, 1, 0, 313])
    parameters['us'] = np.array([2., 0., 1., 
                                 0., 30., 0.])
    parameters['ps'] = np.array([0.8, 0.1, 0.8, 0.1, 313])

    # Get the constraints.
    ulb = np.array([-0.5, -500., -0.5, -500., -0.5, -500.])
    uub = np.array([0.5, 500., 0.5, 500., 0.5, 500.])
    ylb = np.array([-5., 0., 0., -10., 
                    -5., 0., 0., -3., 
                    -1., 0., 0., -10.])
    yub = np.array([5., 1., 1., 10., 
                    5., 1., 1., 3., 
                    1., 1, 1., 10.])
    plb = np.array([-0.1, -0.1, -0.1, -0.1, -8.])
    pub = np.array([0.05, 0.05, 0.05, 0.05, 8.])
    parameters['lb'] = dict(u=ulb, y=ylb, p=plb)
    parameters['ub'] = dict(u=uub, y=yub, p=pub)

    # Get the scaling.
    parameters['uscale'] = 0.5*(parameters['ub']['u'] - parameters['lb']['u'])
    parameters['pscale'] = 0.5*(parameters['ub']['p'] - parameters['lb']['p'])
    parameters['yscale'] = 0.5*(parameters['ub']['y'] - parameters['lb']['y'])

    # Scale the lower and upper bounds for the MPC controller.
    # Scale the bounds.
    parameters['lb']['u'] = parameters['lb']['u']/parameters['uscale']
    parameters['ub']['u'] = parameters['ub']['u']/parameters['uscale']
    parameters['lb']['y'] = parameters['lb']['y']/parameters['yscale']
    parameters['ub']['y'] = parameters['ub']['y']/parameters['yscale']
    parameters['lb']['p'] = parameters['lb']['p']/parameters['pscale']
    parameters['ub']['p'] = parameters['ub']['p']/parameters['pscale']

    # The C matrix for the plant.
    parameters['C'] = np.eye(Nx)

    # The H matrix.
    parameters['H'] = np.zeros((6, Ny))
    parameters['H'][0, 0] = 1.
    parameters['H'][1, 3] = 1.
    parameters['H'][2, 4] = 1.
    parameters['H'][3, 7] = 1.
    parameters['H'][4, 8] = 1.
    parameters['H'][5, 11] = 1.

    # Measurement Noise.
    parameters['Rv'] = 1e-20*np.diag(np.array([1e-4, 1e-6, 1e-6, 1e-4, 
                                               1e-4, 1e-6, 1e-6, 1e-4, 
                                               1e-4, 1e-6, 1e-6, 1e-4]))

    # Return the parameters dict.
    return parameters

def _get_cstrs_rectified_xs(*, parameters):
    """ Get the steady state of the plant."""
    # (xs, us, ps)
    xs = np.zeros((parameters['Nx'], 1))
    us = np.zeros((parameters['Nu'], 1))
    ps = np.zeros((parameters['Np'], 1))
    cstrs_ode = lambda x, u, p: _cstrs_ode(x, u, p, parameters)   
    # Construct the casadi class.
    model = mpc.DiscreteSimulator(cstrs_ode, 
                                  parameters['sample_time'],
                                  [parameters['Nx'], parameters['Nu'], 
                                   parameters['Np']], 
                                  ["x", "u", "p"])
    # steady state of the plant.
    for _ in range(7200):
        xs = model.sim(xs, us, ps)
    # Return the disturbances.
    return parameters['xs'] + xs

def _get_linearized_model(*, parameters):
    """ Return a linear model 
    for use in MPC for the cstrs plant."""
    cstrs_ode = lambda x, u, p: _cstrs_ode(x, u, p, parameters)  
    cstrs_ode_casadi = mpc.getCasadiFunc(cstrs_ode,
                                        [parameters['Nx'], parameters['Nu'],    
                                         parameters['Np']],
                                        ["x", "u", "p"],
                                        funcname="cstrs")
    linear_model = mpc.util.getLinearizedModel(cstrs_ode_casadi,
                                            [np.zeros((parameters['Nx'], 1)), 
                                             np.zeros((parameters['Nu'], 1)), 
                                             np.zeros((parameters['Np'], 1))], 
                                             ["A", "B", "Bp"],
                                             Delta=parameters['sample_time'])
    
    # Do some scaling.
    yscale = parameters['yscale']
    C = np.diag(1/yscale.squeeze()) @ parameters['C']
    # Return the matrices.
    return (linear_model['A'], linear_model['B'], 
            C, linear_model['Bp'])

def _get_cstrs_plant(*, linear, parameters):
    """ Return a Nonlinear Plant Simulator object."""
    if linear:  
        (A, B, C, Bp) = _get_linearized_model(parameters=parameters)
        return LinearPlantSimulator(A=A, B=B, C=C, Bp=Bp,
                                    Rv=parameters['Rv'],
                                    sample_time=parameters['sample_time'],
                                    x0=np.zeros((parameters['Nx'], 1)))
    else:
        # Construct and Return the Plant.
        cstrs_ode = lambda x, u, p: _cstrs_ode(x, u, p, parameters)
        cstrs_measurement = lambda x: _cstrs_measurement(x, parameters)
        return NonlinearPlantSimulator(fxup = cstrs_ode,
                                       hx = cstrs_measurement,
                                       Rv = parameters['Rv'], 
                                       Nx = parameters['Nx'], 
                                       Nu = parameters['Nu'], 
                                       Np = parameters['Np'], 
                                       Ny = parameters['Ny'],
                                       sample_time = parameters['sample_time'], 
                                       x0 = np.zeros((parameters['Nx'], 1)))

def _get_cstrs_mpc_controller(plant, parameters, 
                              z_indices, exp_dist_indices):
    """Construct a linear MPC controller object for the double 
    integrator plant.""" 
    # Get shapes and sizes.
    (A, B, C, Bp) = _get_linearized_model(parameters=parameters)
    (Nx, Nu) = B.shape
    (Ny, _) = C.shape
    H = np.zeros((0, Ny))
    Nd = len(exp_dist_indices)

    # Disturbance model.
    Bd = Bp[:, exp_dist_indices]
    Cd = np.zeros((Ny, Nd))

    # Kalman Filter Parameters.
    Qwx = (1e-16)*np.eye(Nx)
    Qwd = (1e-2)*np.eye(Nd)
    xprior = plant.x[-1]
    dprior = np.zeros((Nd, 1))
    uprev = np.zeros((Nu, 1))
    Rv = 1e+20*np.diag(plant.measurement_noise_std.squeeze())**2

    # Target Selector Parameters.
    Rs = 0*np.eye(Nu)
    Qs = 0*np.eye(Ny)
    Qs[z_indices, z_indices] = 1.
    usp = np.zeros((Nu,1))

    # Regulator parameters.
    Q = 1e+3*(C.T @ C)
    R = 0.1*np.eye(Nu)
    S = 0.1*np.eye(Nu)
    N = 90
    ulb = parameters['lb']['u'][:, np.newaxis]
    uub = parameters['ub']['u'][:, np.newaxis]

    # Return a linear MPC controller instance.
    return LinearMPCController(A=A, B=B, C=C, H=H,
                               Qwx=Qwx, Qwd=Qwd,
                               Rv=Rv, xprior=xprior, dprior=dprior,
                               Rs=Rs, Qs=Qs, Bd=Bd, Cd=Cd, usp=usp, uprev=uprev,
                               Q=Q, R=R, S=S, N=N, ulb=ulb, uub=uub)

def _get_cstrs_offline_simulator(controller, parameters,
                                 z_indices, unexp_z_indices, exp_dist_indices,
                                 Nsim, num_data_gen_task, 
                                 num_process_per_task, 
                                 conservative_factor, seed):
    """Get the Offline Model Simulator for the double integrator example."""
    # Upper and lower bound for the disturbances.
    setpoint_lb = parameters['lb']['y']*conservative_factor
    setpoint_ub = parameters['ub']['y']*conservative_factor
    disturbance_lb = parameters['lb']['p']*conservative_factor
    disturbance_ub = parameters['ub']['p']*conservative_factor

    # Get the setpoints.
    setpoints_z = np.zeros((Nsim, parameters['Ny']))
    setpoints_y = sample_prbs_like(num_change=1250, num_steps=Nsim, 
                             lb=setpoint_lb, ub=setpoint_ub,
                             mean_change=120, sigma_change=2, seed=seed)
    setpoints_z[:, z_indices] = setpoints_y[:, z_indices]
    setpoints_z[:, unexp_z_indices] = np.zeros((Nsim, len(unexp_z_indices)))

    # Get the disturbances. 
    disturbances = sample_prbs_like(num_change=2500, num_steps=Nsim, 
                             lb=disturbance_lb, ub=disturbance_ub,
                             mean_change=60, sigma_change=5, seed=seed+1)
    disturbances = disturbances[:, exp_dist_indices]

    # Return the offline simulator object.
    return  OfflineSimulator(A=controller.A, B=controller.B, C=controller.C, 
                             H=controller.H,
                             Rs=controller.Rs, Qs=controller.Qs, 
                             Bd=controller.Bd,
                             Cd = controller.Cd, usp=controller.usp, uprev=controller.usp,
                             Q=controller.Q, R=controller.R, S=controller.S,
                             ulb=controller.ulb, uub=controller.uub, N=controller.N, 
                             xprior = controller.xprior, setpoints=setpoints_z, 
                             disturbances=disturbances,
                             num_data_gen_task=num_data_gen_task,
                             num_process_per_task=num_process_per_task)

def _get_cstrs_online_test_scenarios(*, Nsim, z_indices, unexp_z_indices,
                                     parameters, exp_dist_indices, seed,
                                     tsteps_steady):
    """ Just generate a setpoint and disturbance sequence for 
    the online comparision of the NN and the optimal MPC controllers 
    Will also use these parameters for testing the online controllers
    """
    # Upper and lower bound for the disturbances.
    setpoint_lb = parameters['lb']['y']
    setpoint_ub = parameters['ub']['y']
    disturbance_lb = parameters['lb']['p']
    disturbance_ub = parameters['ub']['p']
    (Ny, Np) = (parameters['Ny'], parameters['Np'])

    # Setpoints.
    setpoints_z = np.zeros((Nsim, parameters['Ny']))
    setpoints_all = sample_prbs_like(num_change=24, num_steps=Nsim, 
                            lb=setpoint_lb, ub=setpoint_ub,
                            mean_change=180, sigma_change=2, seed=seed)
    setpoints_z[:, z_indices] = setpoints_all[:, z_indices]
    setpoints_z[0:tsteps_steady, :] = np.zeros((tsteps_steady, Ny))
    setpoints_unexp = setpoints_z.copy()
    setpoints_exp = setpoints_z.copy()
    setpoints_exp[:, unexp_z_indices] = np.zeros((Nsim, len(unexp_z_indices)))

    # Disturbances.    
    disturbances = sample_prbs_like(num_change=48, num_steps=Nsim, 
                                lb=disturbance_lb, ub=disturbance_ub,
                                mean_change=90, sigma_change=1, seed=seed+1)
    disturbances[0:tsteps_steady, :] = np.zeros((tsteps_steady, Np))

    scenarios = [(setpoints_exp, disturbances), 
                 (setpoints_unexp, disturbances)]
    # Return the scenarios.
    return scenarios

if __name__ == "__main__":
    # Construct the plant, controller, and the online test parameters.
    z_indices = (0, 3, 4, 7, 8, 11)
    unexp_z_indices = [4]
    exp_dist_indices = (0, 1, 2, 3, 4)
    # Get the plotting parameters.
    cstrs_plant_parameters = _get_cstrs_parameters()
    cstrs_plant_parameters['xs'] = _get_cstrs_rectified_xs(parameters=
                                                        cstrs_plant_parameters)
    cstrs_plant_parameters['exp_dist_indices'] = exp_dist_indices
    cstrs_plant_parameters['z_indices'] = z_indices
    cstrs_plant_parameters['unexp_z_indices'] = unexp_z_indices
    cstrs = _get_cstrs_plant(linear=False, parameters=cstrs_plant_parameters)
    cstrs_mpc_controller = _get_cstrs_mpc_controller(cstrs, 
                                            cstrs_plant_parameters,
                                            z_indices, exp_dist_indices)
    cstrs_us_controller = _get_us_controller(cstrs_mpc_controller)
    cstrs_satdlqr_controller = _get_satdlqr_controller(cstrs_mpc_controller)
    cstrs_sh_controller = _get_short_horizon_controller(cstrs_mpc_controller, 
                                                        N=10)
    cstrs_online_test_scenarios = _get_cstrs_online_test_scenarios(Nsim=4320, 
                                        z_indices=z_indices, unexp_z_indices=unexp_z_indices,
                                        parameters=cstrs_plant_parameters,
                                        exp_dist_indices=exp_dist_indices,
                                        seed=50, tsteps_steady=5)
    cstrs_offline_simulator = _get_cstrs_offline_simulator(cstrs_mpc_controller,
                            cstrs_plant_parameters, z_indices, unexp_z_indices, 
                                            exp_dist_indices, Nsim=150000,
                                            num_data_gen_task=1,
                                            num_process_per_task=1, 
                                            conservative_factor=1.02, seed=1)
    # Create a dictionary with the required data.
    cstrs_parameters = dict(plant=cstrs, mpc=cstrs_mpc_controller,
                            us=cstrs_us_controller,
                            satdlqr=cstrs_satdlqr_controller,
                            short_horizon=cstrs_sh_controller,
                            offline_simulator=cstrs_offline_simulator,
                            online_test_scenarios=cstrs_online_test_scenarios,
                            cstrs_plant_parameters=cstrs_plant_parameters)
    #cstrs_parameters = dict(online_test_scenarios=cstrs_online_test_scenarios, 
    #                        cstrs_plant_parameters=cstrs_plant_parameters)
    # Save data.
    PickleTool.save(data_object=cstrs_parameters, 
                    filename='cstrs_parameters.pickle')