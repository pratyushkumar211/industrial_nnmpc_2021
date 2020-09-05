"""
Script that contains custom python classes to solve
nonlinear tracking MPC problems using Casadi/IPOPT.
Author:Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import time
import numpy as np
import mpctools as mpc
import scipy.linalg

class NonlinearPlantSimulator:
    """Custom class for simulating non-linear plants."""
    def __init__(self, *, fxup, hx, Rv, Nx, Nu, Np, Ny, 
                 sample_time, x0):
        
        # Set attributes.
        self.fxup = mpc.DiscreteSimulator(fxup, sample_time,
                                          [Nx, Nu, Np], ["x", "u", "p"])
        self.hx = mpc.getCasadiFunc(hx, [Nx], ["x"], funcname="hx")
        (self.Nx, self.Nu, self.Ny, self.Np) = (Nx, Nu, Ny, Np)
        self.measurement_noise_std = np.sqrt(np.diag(Rv)[:, np.newaxis])
        self.sample_time = sample_time

        # Create lists to save data.
        self.x = [x0]
        self.u = []
        self.p = []
        self.y = [np.asarray(self.hx(x0)) + 
                 self.measurement_noise_std*np.random.randn(self.Ny, 1)]
        self.t = [0.]

    def step(self, u, p):
        """ Inject the control input into the plant."""
        x = self.fxup.sim(self.x[-1], u, p)[:, np.newaxis]
        y = np.asarray(self.hx(x))
        y = y + self.measurement_noise_std*np.random.randn(self.Ny, 1)
        self._append_data(x, u, p, y)
        return y

    def _append_data(self, x, u, p, y):
        """ Append the data into the lists.
        Used for plotting in the specific subclasses.
        """
        self.x.append(x)
        self.u.append(u)
        self.p.append(p)
        self.y.append(y)
        self.t.append(self.t[-1]+self.sample_time)

class NonlinearMHEEstimator:

    def __init__(self, *, fxuw, hx, N, Nx, Nu, Ny,
                 xprior, u, y, P0inv, Qwinv, Rvinv):
        """ Class to construct and perform state estimation
        using moving horizon estimation. 
        
        Problem setup:
        The current time is T
        Measurement available: u_[T-N:T-1], y_[T-N:T]

        Optimization problem:
        min_{x_[T-N:T]} |x(T-N)-xprior|_{P0inv} + sum{t=T-N to t=N-1} |x(k+1)-x(k)|_{Qwinv} + sum{t=T-N to t=T} |y(k)-h(x(k))|_{Rvinv}

        subject to: x(k+1) = f(x(k), u(k), w(k)), y(k) = h(x) + v(k), k=T-N to T-1
        x is the augmented state.

        xprior is an array of previous smoothed estimates, xhat(k:k) from -T:-1
        y: -T:0
        u: -T:-1
        The constructor solves and gets a smoothed estimate of x at 0.
        """
        self.fxuw = fxuw
        self.hx = hx

        # Penalty matrices.
        self.P0inv = P0inv
        self.Qwinv = Qwinv
        self.Rvinv = Rvinv

        # Sizes.
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny
        self.N = N

        # Create lists for saving data. 
        self.xhat = list(xprior)
        self.y = list(y)
        self.u = list(u)

        # Build the estimator.
        self._setup_moving_horizon_estimator()

    def _setup_moving_horizon_estimator(self):
        """ Construct a MHE solver."""
        N = dict(x=self.Nx, u=self.Nu, y=self.Ny, t=self.N)
        funcargs = dict(f=["x", "u"], h=["x"], l=["w", "v"], lx=["x", "x0bar"])
        l = mpc.getCasadiFunc(self._stage_cost, [N["x"], N["y"]],
                              funcargs["l"])
        lx = mpc.getCasadiFunc(self._prior_cost, [N["x"], N["x"]],
                              funcargs["lx"])
        self.moving_horizon_estimator = mpc.nmhe(f=self.fxuw, h=self.hx, wAdditive=True,
                                                 N=N, l=l, lx=lx, u=self.u, y=self.y,
                                                 funcargs=funcargs, x0bar=self.xhat[0], verbosity=0)
        self.moving_horizon_estimator.solve()
        self.xhat.append(self.moving_horizon_estimator.var["x"][-1])

    def _stage_cost(self, w, v):
        """Stage cost in moving horizon estimation."""
        return mpc.mtimes(w.T, self.Qwinv, w) + mpc.mtimes(v.T, self.Rvinv, v)

    def _prior_cost(self, x, xprior):
        """Prior cost in moving horizon estimation."""
        dx = x - xprior
        return mpc.mtimes(dx.T, self.P0inv, dx)
        
    def solve(self, y, uprev):
        """Use the new data, solve the NLP, and store data.
        At this time:
        xhat: list of length T+1
        y: list of length T+1
        uprev: list of length T 
        """
        #self.moving_horizon_estimator.par["x0bar"] = 
        self.moving_horizon_estimator.par["y"] = [self.y[-self.N:], y]        
        self.moving_horizon_estimator.par["u"] = [self.u[-self.N+1:], uprev]        
        self.moving_horizon_estimator.solve()
        xhat = self.moving_horizon_estimator.var["x"][-1]
        self._append_data(xhat, y, uprev)
        return xhat

    def _append_data(self, xhat, y, uprev):
        """ Append the data to the lists."""
        self.xhat.append(xhat)
        self.y.append(y)
        self.u.append(uprev)

class NonlinearTargetSelector:

    def __init__(self, *, fxud, hxd, Nx, Nu, Nd, Ny, 
                 ys, us, ds, Rs, Qs, ulb, uub):
        """ Class to construct and solve the following 
            NLP target selection problem.
        min_(xs, us) |us - usp|^2_Rs + |h(xs, d) - ysp|^2_Qs
        s.t f(xs, us, d) = xs
            F*h(xs, d) <= f
            E*us <= e

        Construct the class and use the method "solve" 
        for obtaining the solution.
        
        An instance of this class will also
        store the history of the solutions obtained.  
        """
        self.fxud = fxud
        self.hxd = hxd

        # Data for the class.
        self.usp = [us]
        self.ysp = [ys]
        self.ds = [ds]
        self.xs = []
        self.us = []
        self.Rs = Rs
        self.Qs = Qs

        # Sizes.
        self.Nx = Nx
        self.Nu = Nu
        self.Nd = Nd
        self.Ny = Ny

        # Get the hard constraints on inputs and the soft constraints. 
        self.ulb = ulb
        self.uub = uub
        
        # Setup the target selector.
        self._setup_target_selector()
        
    def _setup_target_selector(self):
        """ Construct the target selector object for this class."""
        N = dict(x=self.Nx, u=self.Nu, d=self.Nd, y=self.Ny)
        funcargs = dict(f=["x", "u", "ds"], h=["x", "ds"], phi=["y", "u", "ysp", "usp"])
        extrapar = dict(ysp=self.ysp[-1], usp=self.usp[-1], ds=self.ds[-1])
        phi = mpc.getCasadiFunc(self._stage_cost, [N["y"], N["u"], N["y"], N["u"]],
                                funcargs["phi"])
        lb = dict(u=self.ulb.T)
        ub = dict(u=self.uub.T)
        self.target_selector = mpc.sstarg(f=self.fxud, h=self.hxd, 
                                          phi=phi, N=N, lb=lb, ub=ub,
                                          extrapar=extrapar, funcargs=funcargs, verbosity=0)
        self.target_selector.solve()
        self.xs.append(self.target_selector.var["x"][0])
        self.us.append(self.target_selector.var["u"][0])

    def _stage_cost(self, ys, us, ysp, usp):
        """ The stage cost for the target selector."""
        dy = ys - ysp
        du = us - usp
        return mpc.mtimes(dy.T, self.Qs, dy) + mpc.mtimes(du.T, self.Rs, du)
        
    def solve(self, ysp, ds):
        "Solve the target selector NP, output is (xs, us)."
        self.target_selector.par["ysp"] = ysp
        self.target_selector.par["ds"] = ds
        # Solve and save data.
        self.target_selector.solve()
        xs = self.target_selector.var["xs"][0]
        us = self.target_selector.var["us"][0]
        self._append_data(xs, us, ds, ysp)
        return (xs, us)

    def _append_data(self, xs, us, ds, ysp):
        """ Append the data into the lists.
        Used for plotting in the specific subclasses.
        """
        self.xs.append(xs)
        self.us.append(us)
        self.ds.append(ds)
        self.ysp.append(ysp)

class NonlinearMPCRegulator:

    def __init__(self, *, fxud, hxd,
                 xs, us, ds, 
                 Nx, Nu, Nd,
                 N, Q, R, P, ulb, uub):
        """ Class to construct and solve nonlinear MPC -- Regulation. 
        
        Problem setup:
        The current time is T, we have x (absolute).

        Optimization problem:
        min_{u[0:N-1]} sum_{k=0^k=N-1} [|x(k)-xs|_{Q} + |u(k)-us|_R] + |x(N)-xs|_{P}
        subject to: x(k+1) = f(x(k), u(k), ds), k=0 to N-1
        """
        self.fxud = fxud
        self.hxd = hxd

        # Penalty matrices.
        self.Q = Q
        self.R = R
        self.P = P

        # Sizes.
        self.N = N
        self.Nx = Nx
        self.Nu = Nu
        self.Nd = Nd

        # Create lists for saving data. 
        self.x0 = [xs]
        self.xs = [xs]
        self.us = [us]
        self.ds = [ds]
        self.useq = []

        # Get the hard constraints on inputs and the soft constraints. 
        self.ulb = ulb
        self.uub = uub

        # Build the estimator.
        self._setup_nonlinear_regulator()

    def _setup_nonlinear_regulator(self):
        """ Construct a MHE solver."""
        N = dict(x=self.Nx, u=self.Nu, t=self.N)
        funcargs = dict(f=["x", "u", "ds"], l=["x", "u", "xs", "us"], Pf=["x", "xs"])
        extrapar = dict(xs=self.xs[-1], us=self.us[-1], ds=self.ds[-1])
        l = mpc.getCasadiFunc(self._stage_cost, [self.Nx, self.Nu, self.Nx, self.Nu],
                              funcargs["l"])
        Pf = mpc.getCasadiFunc(self._terminal_cost, [self.Nx, self.Nx],
                              funcargs["Pf"])
        lb = dict(u=self.ulb.T)
        ub = dict(u=self.uub.T)
        self.nonlinear_regulator = mpc.nmpc(f=self.fxud, l=l, Pf=Pf, 
                                            N=N, lb=lb, ub=ub, extrapar=extrapar,
                                            funcargs=funcargs, x0=self.x0[-1].squeeze(axis=-1))
        self.nonlinear_regulator.par["xs"] = self.xs*(self.N + 1)
        self.nonlinear_regulator.par["us"] = self.us*self.N
        self.nonlinear_regulator.par["ds"] = self.ds*self.N
        self.nonlinear_regulator.solve()
        breakpoint()
        self.useq.append(np.asarray(self.nonlinear_regulator.var["u"]))

    def _stage_cost(self, x, u, xs, us):
        """Stage cost in moving horizon estimation."""
        dx = x - xs
        du = u - us
        return mpc.mtimes(dx.T, self.Q, dx) + mpc.mtimes(du.T, self.R, du)

    def _terminal_cost(self, x, xs):
        """Prior cost in moving horizon estimation."""
        dx = x - xs
        return mpc.mtimes(dx.T, self.P, dx)

    def solve(self, x0, xs, us, ds):
        """Setup and the solve the dense QP, output is 
        the first element of the sequence.
        If the problem is reparametrized, go back to original 
        input variable. 
        """
        self.nonlinear_regulator.par["xs"] = [xs]*(self.N + 1)
        self.nonlinear_regulator.par["us"] = [us]*self.N
        self.nonlinear_regulator.par["ds"] = [ds]*self.N
        self.nonlinear_regulator.fixvar("x", 0, x0)
        self.nonlinear_regulator.solve()
        useq = np.asarray(self.nonlinear_regulator.var["u"])
        self._append_data(x0, useq)
        return useq

    def _append_data(self, x0, useq):
        "Append data."
        self.x0.append(x0)
        self.useq.append(useq)

class NonlinearMPCController:
    """ Class that instantiates a NonlinearMHE, 
        a NonlinearTargetSelector, and a NonlinearRegulator classes
        into one and solves tracking MPC problems with 
        linear models.

        fxud is a continous time model.
        hxd is same in both the continous and discrete time.
    """
    def __init__(self, *, fxud, hxd, sample_time,
                 num_rk4_discretization_steps, Nx, Nu, Ny, Nd, 
                 Qwx, Qwd, Rv, Nmhe,
                 xs, us, ds, ys,
                 Rs, Qs,
                 Q, R, P, ulb, uub, Nmpc):
        
        # Save attributes.
        self.fxud = fxud
        self.hxd = hxd

        # Known steady states of the system.
        self.xs = xs
        self.us = us
        self.ds = ds
        self.ys = ys

        # MHE Parameters.
        self.Qwx = Qwx
        self.Qwd = Qwd
        self.Rv = Rv
        self.Nmhe = Nmhe

        # Target selector parameters.
        self.Rs = Rs
        self.Qs = Qs

        # MPC Regulator Parameters.
        self.Q = Q
        self.R = R
        self.P = P
        self.ulb = ulb
        self.uub = uub
        self.Nmpc = Nmpc
        self.uprev = us

        # Sizes.
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny
        self.Nd = Nd

        # Instantiate the required classes. 
        self.filter = NonlinearMPCController.setup_filter(fxud=fxud, hxd=hxd, sample_time=sample_time,
                                                          num_rk4_discretization_steps=num_rk4_discretization_steps, 
                                                          Nmhe=Nmhe, Nx=Nx, Nu=Nu, Ny=Ny, Nd=Nd, 
                                                          Qwx=Qwx, Qwd=Qwd, Rv=Rv, 
                                                          xs=xs, us=us, ds=ds, ys=ys)
        self.target_selector = NonlinearMPCController.setup_target_selector(fxud=fxud, hxd=hxd, sample_time=sample_time,
                                                                            num_rk4_discretization_steps=num_rk4_discretization_steps,
                                                                            Nx=Nx, Nu=Nu, Nd=Nd, Ny=Ny, 
                                                                            ys=ys, us=us, ds=ds, Rs=Rs, Qs=Qs, 
                                                                            ulb=ulb, uub=uub)
        self.regulator = NonlinearMPCController.setup_regulator(fxud=fxud, hxd=hxd, sample_time=sample_time,
                                                                num_rk4_discretization_steps=num_rk4_discretization_steps, 
                                                                Nmpc=Nmpc, Nx=Nx, Nu=Nu, Nd=Nd,
                                                                xs=xs, us=us, ds=ds, 
                                                                Q=Q, R=R, P=P,
                                                                ulb=ulb, uub=ulb)

        # List object to store how much time it takes to solve each MPC problem. 
        self.computation_times = [0.]

    @staticmethod
    def setup_filter(fxud, hxd, sample_time, num_rk4_discretization_steps, 
                     Nmhe, Nx, Nu, Ny, Nd, Qwx, Qwd, Rv, xs, us, ds, ys):
        """ Augment the system with an integrating 
        disturbance and setup the Kalman Filter."""
        (fxuw, hx, P0inv, Qwinv, Rvinv, xprior, u, y) = NonlinearMPCController.get_mhe_models_and_matrices(fxud, hxd, sample_time, 
                                                                                                           num_rk4_discretization_steps,
                                                                                                           Nx, Nu, Nd, Nmhe, 
                                                                                                           Qwx, Qwd, Rv, xs, us, ds, ys)
        return NonlinearMHEEstimator(fxuw=fxuw, hx=hx, N=Nmhe, Nx=Nx+Nd, Nu=Nu, Ny=Ny,
                                     xprior=xprior, u=u, y=y, P0inv=P0inv, Qwinv=Qwinv, Rvinv=Rvinv)

    @staticmethod
    def setup_target_selector(fxud, hxd, sample_time, num_rk4_discretization_steps,
                              Nx, Nu, Nd, Ny, ys, us, ds, Rs, Qs, ulb, uub):
        """ Setup the target selector for the MPC controller."""
        fxud = mpc.getCasadiFunc(fxud, [Nx, Nu, Nd], ["x", "u", "ds"],
                                 rk4=True, Delta=sample_time, M=num_rk4_discretization_steps)
        hxd = mpc.getCasadiFunc(hxd, [Nx, Nd], ["x", "ds"])
        return NonlinearTargetSelector(fxud=fxud, hxd=hxd, Nx=Nx, Nu=Nu, Nd=Nd, Ny=Ny, 
                                        ys=ys, us=us, ds=ds, Rs=Rs, Qs=Qs, ulb=ulb, uub=uub)

    @staticmethod
    def setup_regulator(fxud, hxd, sample_time, num_rk4_discretization_steps, 
                        Nmpc, Nx, Nu, Nd, xs, us, ds, Q, R, P, ulb, uub):
        """ Augment the system for rate of change penalty and 
        build the regulator."""
        fxud = mpc.getCasadiFunc(fxud, [Nx, Nu, Nd], ["x", "u", "ds"],
                                 rk4=True, Delta=sample_time, M=num_rk4_discretization_steps)
        hxd = mpc.getCasadiFunc(hxd, [Nx, Nd], ["x", "ds"])
        return NonlinearMPCRegulator(fxud=fxud, hxd=hxd,
                                     xs=xs, us=us, ds=ds, 
                                     Nx=Nx, Nu=Nu, Nd=Nd,
                                     N=Nmpc, Q=Q, R=R, P=P, ulb=ulb, uub=uub)
    
    @staticmethod
    def get_mhe_models_and_matrices(fxud, hxd, sample_time, num_rk4_discretization_steps, 
                                    Nx, Nu, Nd, Nmhe, Qwx, Qwd, Rv, xs, us, ds, ys):
        """ Get the models, proir estimates and data, and the penalty matrices to setup an MHE solver."""

        # Prior estimates and data.
        xprior = np.concatenate((xs, ds), axis=0)
        xprior = np.repeat(xprior.T, Nmhe, axis=0)
        u = np.repeat(us, Nmhe, axis=0)
        y = np.repeat(ys, Nmhe+1, axis=0)

        # Penalty matrices.
        Qwxinv = np.linalg.inv(Qwx)
        Qwdinv = np.linalg.inv(Qwd)
        Qwinv = scipy.linalg.block_diag(Qwxinv, Qwdinv)
        P0inv = Qwinv
        Rvinv = np.linalg.inv(Rv)

        # Get the augmented models.
        fxuw = mpc.getCasadiFunc(NonlinearMPCController.mhe_state_space_model(fxud, Nx, Nd), [Nx+Nd, Nu], ["x", "u"],
                                                                              rk4=True, Delta=sample_time, M=num_rk4_discretization_steps)
        hx = mpc.getCasadiFunc(NonlinearMPCController.mhe_measurement_model(hxd, Nx), [Nx+Nd], ["x"])
        # Return the required quantities for MHE.
        return (fxuw, hx, P0inv, Qwinv, Rvinv, xprior, u, y)

    @staticmethod
    def mhe_state_space_model(fxud, Nx, Nd):
        """Augmented state-space model for moving horizon estimation."""
        return lambda x, u : np.concatenate((fxud(x[0:Nx], u, x[Nx:]), np.zeros((Nd,))), axis=0)
    
    @staticmethod
    def mhe_measurement_model(hxd, Nx):
        """Augmented measurement model for moving horizon estimation."""
        return lambda x : hxd(x[0:Nx], x[Nx:])

    def control_law(self, ysp, y):
        """
        Takes the measurement and the previous control input
        and compute the current control input.
        """
        tstart = time.time()
        (xhat, ds) =  NonlinearMPCController.get_state_estimates(self.filter, y, self.uprev, self.Nx)
        (xs, us) = NonlinearMPCController.get_target_pair(self.target_selector, ysp, ds)
        self.uprev = NonlinearMPCController.get_control_input(self.regulator, xhat, xs, us, ds)
        tend = time.time()
        self.computation_times.append(tend - tstart)
        return self.uprev
    
    @staticmethod
    def get_state_estimates(filter, y, uprev, Nx):
        """Use the filter object to perform state estimation."""
        return np.split(filter.solve(y, uprev), [Nx])

    @staticmethod
    def get_target_pair(target_selector, ysp, ds):
        """Use the target selector object to 
        compute the targets."""
        return target_selector.solve(ysp, ds)

    @staticmethod
    def get_control_input(regulator, x, xs, us, ds):
        """ Use the nonlinear regulator to solve the.""" 
        return regulator.solve(x, xs, us, ds)[0:1, :]