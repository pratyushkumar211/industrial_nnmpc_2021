""" Module for solving linear MPC problems
    and offline data generation for neural network training."""

import sys
import time
import os
import concurrent.futures
import multiprocessing
import numpy as np
import cvxopt as cvx
import scipy.linalg
import itertools
from python_utils import H5pyTool

def array_to_matrix(*arrays):
    """Convert nummpy arrays to cvxopt matrices."""
    matrices = []
    for array in arrays:
        matrices.append(cvx.matrix(array))
    return tuple(matrices)

def dlqr(A,B,Q,R,M=None):
    """
    Get the discrete-time LQR for the given system.
    Stage costs are
        x'Qx + 2*x'Mu + u'Ru
    with M = 0 if not provided.
    """
    # For M != 0, we can simply redefine A and Q to give a problem with M = 0.
    if M is not None:
        RinvMT = scipy.linalg.solve(R,M.T)
        Atilde = A - B.dot(RinvMT)
        Qtilde = Q - M.dot(RinvMT)
    else:
        Atilde = A
        Qtilde = Q
        M = np.zeros(B.shape)
    Pi = scipy.linalg.solve_discrete_are(Atilde,B,Qtilde,R)
    K = -scipy.linalg.solve(B.T.dot(Pi).dot(B) + R, B.T.dot(Pi).dot(A) + M.T)
    return (K, Pi)

def dlqe(A,C,Q,R):
    """
    Get the discrete-time Kalman filter for the given system.
    """
    P = scipy.linalg.solve_discrete_are(A.T,C.T,Q,R)
    L = scipy.linalg.solve(C.dot(P).dot(C.T) + R, C.dot(P)).T
    return (L, P)

def c2d(A, B, sample_time):
    """ Custom c2d function for linear systems."""
    
    # First construct the incumbent matrix
    # to take the exponential.
    (Nx, Nu) = B.shape
    M1 = np.concatenate((A, B), axis=1)
    M2 = np.zeros((Nu, Nx+Nu))
    M = np.concatenate((M1, M2), axis=0)
    Mexp = scipy.linalg.expm(M*sample_time)

    # Return the extracted matrices.
    Ad = Mexp[:Nx, :Nx]
    Bd = Mexp[:Nx, -Nu:]
    return (Ad, Bd)

def _eigval_eigvec_test(X,Y):
    """Return True if an eigenvector of X corresponding to 
    an eigenvalue of magnitude greater than or equal to 1
    is not in the nullspace of Y.
    Else Return False."""
    (eigvals, eigvecs) = np.linalg.eig(X)
    eigvecs = eigvecs[:, np.absolute(eigvals)>=1.]
    for eigvec in eigvecs.T:
        if np.linalg.norm(Y @ eigvec)<=1e-8:
            return False
    else:
        return True

def assert_detectable(A, C):
    """Assert if the provided (A, C) pair is detectable."""
    assert _eigval_eigvec_test(A, C)

def assert_stabilizable(A, B):
    """Assert if the provided (A, B) pair is stabilizable."""
    assert _eigval_eigvec_test(A.T, B.T)

class LinearPlantSimulator:
    """Class for simulating linear plants."""
    def __init__(self, *, A, B, C, Bp, Rv, sample_time, x0):

        # Set attributes.
        self.A = A
        self.B = B
        self.C = C
        self.Bp = Bp

        # Plant shapes and sizes.
        self.Nx = A.shape[0]
        self.Nu = B.shape[1]
        self.Ny = C.shape[0]

        # Measurement noise and sample time.
        self.measurement_noise_std = np.sqrt(np.diag(Rv)[:, np.newaxis])
        self.sample_time = sample_time

        # Create lists to save data.
        self.x = [x0]
        self.u = []
        self.p = []
        self.v = [self.measurement_noise_std*np.random.randn(self.Ny, 1)]
        self.y = [self.C @ x0 + self.v[-1]]
        self.t = [0.]

    def step(self, u, p):
        """ Inject the control input into the plant."""
        x = self.A @ self.x[-1] + self.B @ u + self.Bp @ p
        v = self.measurement_noise_std*np.random.randn(self.Ny, 1)
        y = self.C @ x + v
        self._append_data(x, u, p, v, y)
        return y

    def _append_data(self, x, u, p, v, y):
        """ Append the data into the lists.
        Used for plotting in the specific subclasses.
        """
        self.x.append(x)
        self.u.append(u)
        self.p.append(p)
        self.v.append(v)
        self.y.append(y)
        self.t.append(self.t[-1]+self.sample_time)

class KalmanFilter:

    def __init__(self, *, A, B, C, Qw, Rv, xprior):
        """ Class to construct and perform state estimation
        using Kalman Filtering.
        """

        # Store the matrices.
        self.A = A
        self.B = B 
        self.C = C
        self.Qw = Qw
        self.Rv = Rv
        
        # Compute the kalman filter gain.
        self._computeFilter()
        
        # Create lists for saving data. 
        self.xhat = [xprior]
        self.xhat_pred = []
        self.y = []
        self.uprev = []

    def _computeFilter(self):
        "Solve the DARE to compute the optimal L."
        (self.L, _) = dlqe(self.A, self.C, self.Qw, self.Rv) 

    def solve(self, y, uprev):
        """ Take a new measurement and do 
            the prediction and filtering steps."""
        xhat = self.xhat[-1]
        xhat_pred = self.A @ xhat + self.B @ uprev
        xhat = xhat_pred + self.L @ (y - self.C @ xhat_pred)
        # Save data.
        self._save_data(xhat, xhat_pred, y, uprev)
        return xhat
        
    def _save_data(self, xhat, xhat_pred, y, uprev):
        """ Save the state estimates,
            Can be used for plotting later."""
        self.xhat.append(xhat)
        self.xhat_pred.append(xhat_pred)
        self.y.append(y)
        self.uprev.append(uprev)
        
class TargetSelector:

    def __init__(self, *, A, B, C, H, Bd, Cd, usp, 
                 Rs, Qs, ulb, uub, ylb=None, yub=None):
        """Class to construct and solve the following 
        target selector problem.
        min_(xs, us) |us - usp|^2_Rs + |C*xs + Cd*dhats - ysp|^2_Qs
        s.t [I-A, -B;HC, 0][xs;us] = [Bd*dhats;H*(ysp-Cd*dhats)]
            F*C*xs <= f - F*Cd*dhats
            E*us <= e

        Construct the class and use the method "solve" 
        for obtaining the solution.
        
        An instance of this class will also
        store the history of the solutions obtained.  
        """
        
        # Store the matrices.
        self.A = A
        self.B = B
        self.C = C
        self.H = H
        self.Bd = Bd
        self.Cd = Cd
        self.Rs = Rs
        self.Qs = Qs

        # Get the store the sizes.
        self.Nx = A.shape[0]
        self.Nu = B.shape[1]
        self.Ny = C.shape[0]
        self.Nd = Bd.shape[1]
        self.Nz = H.shape[0]

        # Data for the class.
        self.usp = usp
        self.ysp = []
        self.dhats = []
        self.xs = []
        self.us = []

        # Get the input and output constraints.
        self.ulb = ulb
        self.uub = uub
        self.ylb = ylb
        self.yub = yub

        # Setup the fixed matrices.
        self._setup_fixed_matrices()
    
    def _setup_fixed_matrices(self):
        """ Setup the matrices which don't change in 
            an on-line simulation.
            """
        Nx = self.Nx
        Nu = self.Nu
        Ny = self.Ny
        Nz = self.Nz

        # Get the (e, E, f, F, G) matrices.
        # Also get the h matrix if we have only input constraints.
        E = np.concatenate((np.eye(Nu), -np.eye(Nu)), axis=0)
        self.F = np.concatenate((np.eye(Ny), -np.eye(Ny)), axis=0)
        if self.ylb is not None and self.yub is not None:
            G1 = np.concatenate((self.F @ self.C, np.zeros((2*Ny, Nu))), axis=1)
            G2 = np.concatenate((np.zeros((2*Nu, Nx)), E), axis=1)
            self.G = np.concatenate((G1, G2), axis=0)
            self.f = np.concatenate((self.yub, -self.ylb), axis=0)
            self.e = np.concatenate((self.uub, -self.ulb), axis=0)
            self.h = None
        else:
            self.G = np.concatenate((np.zeros((2*Nu, Nx)), E), axis=1)
            self.h = np.concatenate((self.uub, -self.ulb), axis=0)

        # Get the equality constraint matrices.
        A11 = np.eye(Nx) - self.A
        A12 = -self.B
        A21 = self.H @ self.C 
        A22 = np.zeros((Nz, Nu))
        tA1 = np.concatenate((A11, A12), axis=1)
        tA2 = np.concatenate((A21, A22), axis=1)
        self.tA = np.concatenate((tA1, tA2), axis=0)
        b11 = np.zeros((Nx, Ny))
        b12 = self.Bd
        b21 = self.H
        b22 = -(self.H @ self.Cd)
        tb1 = np.concatenate((b11, b12), axis=1)
        tb2 = np.concatenate((b21, b22), axis=1)
        self.tb = np.concatenate((tb1, tb2), axis=0)

        # Get the penalty matrix P.
        P11 = self.C.T @ (self.Qs @ self.C)
        P22 = self.Rs
        P1 = np.concatenate((P11, np.zeros((self.Nx, self.Nu))), axis=1)
        P2 = np.concatenate((np.zeros((self.Nu, self.Nx)), P22), axis=1)
        self.P = np.concatenate((P1, P2), axis=0)
        
    def _setup_changing_matrices(self, ysp, dhats):
        """ Get the matrices which change in real-time."""
        # Get the q
        q1 = self.C.T @ (self.Qs @ (ysp - self.Cd @ dhats))
        q2 = self.Rs @ self.usp
        q = np.concatenate((-q1, -q2), axis=0)

        # Get the h.
        if self.h is None: 
            h1 = np.concatenate((self.yub, -self.ylb), axis=0)
            h1 = h1 - self.F @ (self.Cd @ dhats)
            h2 = np.concatenate((self.uub, -self.ulb), axis=0)
            h = np.concatenate((h1, h2), axis=0)
        else:
            h = self.h

        # Get the b.
        b = self.tb @ np.concatenate((ysp, dhats), axis=0)

        # Return.
        return (q, h, b)

    def solve(self, ysp, dhats):
        "Solve the target selector QP, output is the tuple (xs, us)."

        # Get the matrices for the QP which depend of ysp and dhat
        (q, h, b) = self._setup_changing_matrices(ysp, dhats)
        # Solve and save data.
        solution = cvx.solvers.qp(*array_to_matrix(self.P, q, self.G, 
                                                   h, self.tA, b))
        (xs, us) = np.split(np.asarray(solution['x']), [self.Nx])
        # Save Data.
        self._save_data(xs, us, ysp, dhats)

        # Return the solution.
        return (xs, us)

    def _save_data(self, xs, us, ysp, dhats):
        """ Save the state estimates,
            Can be used for plotting later."""
        self.xs.append(xs)
        self.us.append(us)
        self.ysp.append(ysp)
        self.dhats.append(dhats)

class DenseQPRegulator:
    """ Class to construct and solve the regulator QP
        using the dense formulation. 

        The problem:
        V(x0, \mathbf{u}) = (1/2) \sum_{k=0}^{N-1} (x(k)'Qx(k) + u(k)'Ru(k) + 2x(k)'Mu(k)) + (1/2)x(N)'Pfx(N)
        subject to:
            x(k+1) = Ax(k) + Bu(k)
            ulb <= u(k) <= uub
            Su'x(N) == 0

        This class eliminates all the states
        from the set of decision variables and 
        solves a dense formulation of the QP. 
        This formulation takes polynomial complexity 
        in the state, input, and horizon length dimensions 
        to solve the QP in real-time.
    """
    def __init__(self, *, A, B, Q, R, M, N, ulb, uub):

        # Set attributes.
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.M = M
        self.N = N
        self.ulb = ulb
        self.uub = uub

        # Set sizes.
        self.Nx = A.shape[0]
        self.Nu = B.shape[1]

        # Get the LQR gain, only used for reparameterization for the dense QP.
        (self.Krep, self.Pf) = dlqr(A, B, Q, R, M)

        # First get the terminal equality constraint, and the cost-to-go.
        self._reparameterize()
        self._setup_fixed_matrices()

        # Create lists to save data.
        self.x0 = []
        self.useq = []

    def _reparameterize(self):
        """
        Reparameterize A, B, Q, R, M, and G, h.
        A = A+BK, B = B, Q = Q+K'RK+MK+K'M', R=R
        M = K'R + M
        Pf is the solution of the Riccati Eqn using the 
        new parametrized matrices. 
        """
        (eigvals, _) = np.linalg.eig(self.A)
        if any(np.absolute(eigvals)>=1.):
            self.A = self.A + self.B @ self.Krep
            self.Q = self.Q + self.Krep.T @ (self.R @ self.Krep)
            self.Q = self.Q + self.M @ self.Krep + self.Krep.T @ self.M.T
            self.M = self.Krep.T @ self.R + self.M
            self.reparameterize = True
        else:
            self.reparameterize = False 

    def _setup_fixed_matrices(self):
        """" Setup the fixed matrices which don't change in 
             real-time.
             Finally the QP should be in this format.
             min_x  (1/2)x'Px + q'x 
             s.t     Gx <= h
                     Ax = b
        """
        (self.tA, self.tB) = self._get_tA_tB()
        (tQ, tR, tM, self.tE, self.tK) = self._get_tQ_tR_tM_tE_tK()
        (self.P, self.tq) = self._get_P_tq(self.tA, self.tB, tQ, tR, tM)
        self.G = self._getG(self.tB, self.tE, self.tK)

    def _get_tA_tB(self):
        """ Get the Atilde and Btilde.
        N is the number of state outputs is asked for. 
        
        x = tA*x0 + tB*u
        x = (N+1)n, u=Nm
        tA = (N+1)n*n, tB = (N+1)n*Nm

        tA = [I;
              A;
              A^2;
              .;
              A^N]
        tB = [0, 0, 0, ..;
              B, 0, 0, ...;
              AB, B, 0, ...; 
              ...;
              A^(N-1)B, A^(N-2)B, ..., B]
        """
        tA = np.concatenate([np.linalg.matrix_power(self.A, i) 
                            for i in range(self.N+1)])
        tB = np.concatenate([self._get_tB_row(i) for i in range(self.N+1)])
        return (tA, tB)
    
    def _get_tB_row(self, i):
        """ Returns the ith row of tB."""
        (Nx, Nu) = self.B.shape
        tBi = [np.linalg.matrix_power(self.A, i-j-1) @ self.B 
                if j<i 
                else np.zeros((Nx, Nu)) 
                for j in range(self.N)]
        return np.concatenate(tBi, axis=1)

    def _get_tQ_tR_tM_tE_tK(self):
        """ Get the block diagonal matrices for dense QP. 
        tQ = [Q, 0, 0, ...
              0, Q, 0, ...
              0, 0, Q, ...
              ...
              0, 0, 0, ..., Pf]
        tR = [R, 0, 0, ...
              0, R, 0, ...
              0, 0, R, ...
              ...
              0, 0, 0, ... R]
        tM = [M, 0, 0, ...
              0, M, 0, ...
              0, 0, M, ...
              ...
              0, 0, 0, ..., M
              0, 0, 0, ......]
        tK = [K, 0, 0, ......,
              0, K, 0, ......,
              0, 0, K, 0, ...,
              ...............,
              0, 0, 0, ......K]
        """
        tQ = scipy.linalg.block_diag(*[self.Q if i<self.N else self.Pf
                                    for i in range(self.N+1)])
        tR = scipy.linalg.block_diag(*[self.R for _ in range(self.N)])
        tM = scipy.linalg.block_diag(*[self.M for _ in range(self.N)])
        tM = np.concatenate((tM, np.zeros((self.Nx, self.N*self.Nu))), axis=0)
        E = np.concatenate((np.eye(self.Nu), -np.eye(self.Nu)), axis=0)
        tE = scipy.linalg.block_diag(*[E for _ in range(self.N)])
        if self.reparameterize:
            tK = scipy.linalg.block_diag(*[self.Krep for _ in range(self.N)])
        else:
            tK = None
        return (tQ, tR, tM, tE, tK) 

    def _get_P_tq(self, tA, tB, tQ, tR, tM):
        """ Get the penalites for solving the QP.
            P = tB'*tQ*tB + tR + tB'*tM + tM'*tB 
            tq = (tB'*Q + M)*tA
        """
        P = tB.T @ (tQ @ tB) + tR + tB.T @ tM + tM.T @ tB
        tq = (tB.T @ tQ  + tM.T) @ tA
        return (P, tq) 

    def _getG(self, tB, tE, tK):
        """ Get the inequality matrix for the dense QP."""
        if self.reparameterize:
            G = tE @ (tK @ tB[0:self.N*self.Nx, :]) + tE
        else:
            G = tE
        return G

    def _get_h(self, x0):
        """ Get the RHS of the equality constraint."""
        # Get te.
        e = np.concatenate((self.uub, -self.ulb), axis=0)
        te = np.concatenate([e for _ in range(self.N)])
        if self.reparameterize:
            h = te - self.tE @ (self.tK @ (self.tA[0:self.N*self.Nx, :] @ x0)) 
        else:
            h = te 
        return h

    def solve(self, x0):
        """Setup and the solve the dense QP, output is 
        the first element of the sequence.
        If the problem is reparametrized, go back to original 
        input variable. 
        """
        # Update the RHS of the constraint, becuase they change.
        h = self._get_h(x0)
        solution = cvx.solvers.qp(*array_to_matrix(self.P, self.tq @ x0, 
                                                   self.G, h))
        # Extract the inputs.
        useq = np.asarray(solution['x'])
        if self.reparameterize:
            useq = self.tK @ ((self.tA[:self.N*self.Nx, :] @ x0) + 
                              (self.tB[:self.N*self.Nx, :] @ useq)) + useq
        self._save_data(x0, useq)
        # Return the optimized input sequence.
        return useq

    def _save_data(self, x0, useq):
        # Save data. 
        self.x0.append(x0)
        self.useq.append(useq)

class LinearMPCController:
    """Class that instantiates the KalmanFilter, 
    the TargetSelector, and the MPCRegulator classes
    into one and solves tracking MPC problems with 
    linear models.
    """
    def __init__(self, *, A, B, C, H,
                 Qwx, Qwd, Rv, xprior, dprior,
                 Rs, Qs, Bd, Cd, usp, uprev,
                 Q, R, S, ulb, uub, N):
        
        # Save attributes.
        self.A = A
        self.B = B
        self.C = C
        self.H = H
        self.Qwx = Qwx
        self.Qwd = Qwd
        self.Rv = Rv
        self.xprior = xprior
        self.dprior = dprior
        self.Rs = Rs
        self.Qs = Qs
        self.Bd = Bd
        self.Cd = Cd
        self.usp = usp
        self.uprev = uprev
        self.useq = np.tile(uprev, (N, 1))
        self.Q = Q
        self.R = R
        self.S = S
        self.ulb = ulb
        self.uub = uub
        self.N = N

        # Sizes.
        self.Nx = A.shape[0]
        self.Nu = B.shape[1]
        self.Ny = C.shape[0]
        self.Nd = Bd.shape[1]

        # Instantiate the required classes. 
        self.filter = LinearMPCController.setup_filter(A=A, B=B, C=C, Bd=Bd, 
                                                       Cd=Cd,
                                                       Qwx=Qwx, Qwd=Qwd, Rv=Rv, 
                                                       xprior=xprior, dprior=dprior)
        self.target_selector = LinearMPCController.setup_target_selector(A=A, 
                                                        B=B, C=C, H=H,
                                                        Bd=Bd, Cd=Cd, 
                                                        usp=usp, Qs=Qs, Rs=Rs, 
                                                        ulb=ulb, uub=uub)
        self.regulator = LinearMPCController.setup_regulator(A=A, B=B, Q=Q, 
                                                        R=R, S=S, 
                                                        N=N, ulb=ulb, uub=uub)

        # List object to store the average stage 
        # costs and the average computation times.
        aug_mats = LinearMPCController.get_augmented_matrices_for_regulator(A,  
                                                                    B, Q, R, S)
        (_, _, self.Qaug, self.Raug, self.Maug) = aug_mats
        self.average_stage_costs = [np.zeros((1, 1))]
        self.computation_times = []

    @staticmethod
    def setup_filter(A, B, C, Bd, Cd, Qwx, Qwd, Rv, xprior, dprior):
        """ Augment the system with an integrating 
        disturbance and setup the Kalman Filter."""
        (Aaug, Baug, Caug, Qwaug) = LinearMPCController.get_augmented_matrices_for_filter(A, B, C, Bd, Cd, Qwx, Qwd)
        return KalmanFilter(A=Aaug, B=Baug, C=Caug, Qw=Qwaug, Rv=Rv,
                            xprior = np.concatenate((xprior, dprior)))
    
    @staticmethod
    def setup_target_selector(A, B, C, H, Bd, Cd, usp, Qs, Rs, ulb, uub):
        """ Setup the target selector for the MPC controller."""
        return TargetSelector(A=A, B=B, C=C, H=H, Bd=Bd, Cd=Cd, 
                              usp=usp, Rs=Rs, Qs=Qs, ulb=ulb, uub=uub)
    
    @staticmethod
    def setup_regulator(A, B, Q, R, S, N, ulb, uub):
        """ Augment the system for rate of change penalty and 
        build the regulator."""
        aug_mats = LinearMPCController.get_augmented_matrices_for_regulator(A,  
                                                                    B, Q, R, S)
        (Aaug, Baug, Qaug, Raug, Maug) = aug_mats
        return DenseQPRegulator(A=Aaug, B=Baug, Q=Qaug, R=Raug, 
                                N=N, M=Maug, ulb=ulb, uub=uub)
    
    @staticmethod
    def get_augmented_matrices_for_filter(A, B, C, Bd, Cd, Qwx, Qwd):
        """ Get the Augmented A, B, C, and the noise covariance matrix."""
        Nx = A.shape[0]
        Nu = B.shape[1]
        Nd = Bd.shape[1]
        # Augmented A.
        Aaug1 = np.concatenate((A, Bd), axis=1)
        Aaug2 = np.concatenate((np.zeros((Nd, Nx)), np.eye(Nd)), axis=1)
        Aaug = np.concatenate((Aaug1, Aaug2), axis=0)
        # Augmented B.
        Baug = np.concatenate((B, np.zeros((Nd, Nu))), axis=0)
        # Augmented C.
        Caug = np.concatenate((C, Cd), axis=1)
        # Augmented Noise Covariance.
        Qwaug = scipy.linalg.block_diag(Qwx, Qwd)
        # Check that the augmented model is detectable. 
        assert_detectable(Aaug, Caug)
        return (Aaug, Baug, Caug, Qwaug)
    
    @staticmethod
    def get_augmented_matrices_for_regulator(A, B, Q, R, S):
        """ Get the Augmented A, B, C, and the noise covariance matrix."""
        # Get the shapes.
        Nx = A.shape[0]
        Nu = B.shape[1]
        # Augmented A.
        Aaug1 = np.concatenate((A, np.zeros((Nx, Nu))), axis=1)
        Aaug2 = np.zeros((Nu, Nx+Nu))
        Aaug = np.concatenate((Aaug1, Aaug2), axis=0)
        # Augmented B.
        Baug = np.concatenate((B, np.eye((Nu))), axis=0)
        # Augmented Q.
        Qaug = scipy.linalg.block_diag(Q, S)
        # Augmented R.
        Raug = R + S
        # Augmented M.
        Maug = np.concatenate((np.zeros((Nx, Nu)), -S), axis=0)
        return (Aaug, Baug, Qaug, Raug, Maug)

    def control_law(self, ysp, y):
        """
        Takes the measurement, the previous control input,
        and compute the current control input.

        Count times only for solving the regulator QP.
        """
        (xhat, dhat) =  LinearMPCController.get_state_estimates(self.filter, y, 
                                                            self.uprev, self.Nx)
        (xs, us) = LinearMPCController.get_target_pair(self.target_selector, 
                                                       ysp, dhat)
        tstart = time.time()
        self.useq = LinearMPCController.get_control_sequence(self.regulator, 
                                                    xhat, self.uprev, xs, us,
                                                    self.ulb, self.uub)
        tend = time.time()
        avg_ell = LinearMPCController.get_updated_average_stage_cost(xhat, 
                    self.uprev, xs, us, self.useq[0:self.Nu, :], 
                    self.Qaug, self.Raug, self.Maug, 
                    self.average_stage_costs[-1], len(self.average_stage_costs))
        self.average_stage_costs.append(avg_ell)
        self.uprev = self.useq[0:self.Nu, :]
        self.computation_times.append(tend - tstart)
        return self.uprev
    
    @staticmethod
    def get_state_estimates(filter, y, uprev, Nx):
        """Use the filter object to perform state estimation."""
        return np.split(filter.solve(y, uprev), [Nx])

    @staticmethod
    def get_target_pair(target_selector, ysp, dhat):
        """ Use the target selector object to 
            compute the targets."""
        return target_selector.solve(ysp, dhat)

    @staticmethod
    def get_control_sequence(regulator, x, uprev, xs, us, ulb, uub):
        # Change the constraints of the regulator. 
        regulator.ulb = ulb - us
        regulator.uub = uub - us
        # x0 in deviation from the steady state.
        x0 = np.concatenate((x-xs, uprev-us))
        return regulator.solve(x0) + np.tile(us, (regulator.N, 1))

    @staticmethod
    def get_updated_average_stage_cost(x, uprev, xs, us, u, 
                                       Qaug, Raug, Maug, 
                                       average_stage_cost, time_index):
        # Get the augmented state and compute the stage cost.
        x = np.concatenate((x-xs, uprev-us), axis=0)
        u = u - us
        stage_cost = x.T @ (Qaug @ x) + u.T @ (Raug @ u) 
        stage_cost = stage_cost + x.T @ (Maug @ u) + u.T @ (Maug.T @ x)
        # x0 in deviation from the steady state.
        return (average_stage_cost*(time_index-1) + stage_cost)/time_index

def online_simulation(plant, controller, *, setpoints=None, 
                        disturbances=None, Nsim=None, stdout_filename=None):
    """ Run the online simulation for the plant using the controller.
        Returns the same plant object which has the closed-loop data
        stored inside. 
    """
    sys.stdout = open(stdout_filename, 'w')
    measurement = plant.y[0] # Get the latest plant measurement.
    setpoints = setpoints[..., np.newaxis]
    disturbances = disturbances[..., np.newaxis] 
    for (setpoint, disturbance, i) in zip(setpoints, disturbances, range(Nsim)):
        print("Simulation Step:" + f"{i}")
        control_input = controller.control_law(setpoint, measurement)
        print("Computation time:" + str(controller.computation_times[-1]))
        measurement = plant.step(control_input, disturbance)
    return plant

class OfflineSimulator:
    """Class that instantiates
    the TargetSelector and the MPCRegulator classes
    into one and performs simulations offline to generate
    data for neural network training.
    """
    def __init__(self, *, A, B, C, H,
                 Rs, Qs, Bd, Cd, usp, uprev,
                 Q, R, S, ulb, uub, N, xprior,
                 setpoints, disturbances, 
                 num_data_gen_task, num_process_per_task):
        
        # Save attributes.
        self.A = A
        self.B = B
        self.C = C
        self.H = H
        self.Rs = Rs
        self.Qs = Qs
        self.Bd = Bd
        self.Cd = Cd
        self.usp = usp
        self.Q = Q
        self.R = R
        self.S = S
        self.ulb = ulb
        self.uub = uub
        self.N = N
        self.num_data_gen_task = num_data_gen_task
        self.num_process_per_task = num_process_per_task

        # Sizes.
        self.Nx = A.shape[0]
        self.Nu = B.shape[1]
        self.Ny = C.shape[0]
        self.Nd = Bd.shape[1]

        # For the regulator and the target selector.
        self.x0 = xprior
        self.uprev0 = uprev

        # Set up the regulators and target selectors here for parallel processing.
        (self.regulators, self.target_selectors) = ([], [])
        for _ in range(num_process_per_task):
            (regulator, target_selector) = self._get_regulator_target_selector()
            self.regulators.append(regulator)
            self.target_selectors.append(target_selector)
        # Get the setpoints and disturbances as lists.
        scenario_lists = self._split_scenarios(setpoints=setpoints, 
                                               disturbances=disturbances)
        (self.setpoints, self.disturbances) = scenario_lists

    def _get_regulator_target_selector(self):
        """ Get a regulator and a target selector."""
        target_selector = LinearMPCController.setup_target_selector(A=self.A, 
                                    B=self.B, C=self.C, H=self.H, 
                                    Bd=self.Bd, Cd=self.Cd, 
                                    usp=self.usp, Qs=self.Qs, Rs=self.Rs, 
                                    ulb=self.ulb, uub=self.uub)
        regulator = LinearMPCController.setup_regulator(A=self.A, B=self.B, 
                                    Q=self.Q, R=self.R, S=self.S, 
                                    N=self.N, ulb=self.ulb, 
                                    uub=self.uub)
        # Return the objects.
        return (regulator, target_selector)

    def _split_scenarios(self, *, setpoints, disturbances):
        """ Split the setpoint and disturbance signals for parallel processing."""
        setpoint_splitted = []
        disturbances_splitted = []
        num_data_gen_task = self.num_data_gen_task
        num_process_per_task = self.num_process_per_task
        num_processes = num_data_gen_task*num_process_per_task
        Nsim_each_process = int(setpoints.shape[0]/(num_processes))
        for process_number in range(num_processes):
            setpoint = setpoints[process_number*Nsim_each_process:(process_number+1)*Nsim_each_process, :]
            disturbance = disturbances[process_number*Nsim_each_process:(process_number+1)*Nsim_each_process, :]
            setpoint_splitted.append(setpoint)
            disturbances_splitted.append(disturbance)
        setpoints = [setpoint_splitted[task*num_process_per_task:(task+1)*num_process_per_task] for task in range(num_data_gen_task)]
        disturbances = [disturbances_splitted[task*num_process_per_task:(task+1)*num_process_per_task] for task in range(num_data_gen_task)]
        return (setpoints, disturbances)

    def generate_data(self, *, task_number, data_filename, stdout_filename):
        """ Start separate processes and generate the data."""
        sys.stdout = open(stdout_filename, 'w')
        processes = []
        process_number = 0
        for (setpoint, disturbance, regulator, target_selector) in zip(
            self.setpoints[task_number], self.disturbances[task_number], 
            self.regulators, self.target_selectors):
            process_arguments = (task_number, process_number, data_filename, 
                                 self.x0, self.uprev0, self.A, self.B, self.Bd,
                                 regulator, self.ulb, self.uub, target_selector, setpoint, disturbance)
            if self.num_process_per_task == 1:
                simulate_offline(*process_arguments)
            else:
                process = multiprocessing.Process(target=simulate_offline,  
                                                  args=process_arguments)
                process.start()
                processes.append(process)
                process_number +=1
        # Wait for all the processes to finish.
        if self.num_process_per_task !=1:
            for process in processes:
                process.join()

def simulate_offline(task_number, process_number, data_filename, x0, 
                     uprev0, A, B, Bd,
                     regulator, ulb, uub, target_selector, 
                     setpoints, disturbances):
    """The simulate function. 
    CVX, output of the simulation to a file. 
    """
    # Get some parameters for the simulation.
    tstart_data_gen = time.time()
    (_, Nu) = B.shape
    xt = x0
    uprevt = uprev0
    (x, uprev, xs, us, u) = ([xt], [uprevt], [], [], [])
    setpoints = setpoints[:, :, np.newaxis]
    disturbances = disturbances[:, :, np.newaxis]
    Nsim = setpoints.shape[0]

    # Start the data generation loop.
    for (setpoint, disturbance, i) in zip(setpoints, disturbances, range(Nsim)):
        # Print some data to the text.
        print("Process:" + f"{process_number}")
        print("Simulation Step:" + f"{i}")
        # Get the control input.
        tstart = time.time()
        (xst, ust) = LinearMPCController.get_target_pair(target_selector, 
                                                         setpoint, disturbance)
        useqt = LinearMPCController.get_control_sequence(regulator, xt, uprevt, 
                                                         xst, ust, 
                                                         ulb, uub)
        ut = useqt[0:Nu, :]
        tend = time.time()
        print("Computation time:" + str(tend-tstart))
        # Advance the model and append the data point.
        xt = A @ xt + B @ ut + Bd @ disturbance
        uprevt = ut
        x.append(xt)
        uprev.append(uprevt)
        xs.append(xst)
        us.append(ust)
        u.append(ut)  
    # Now when the loop is over.
    x = np.asarray(x[:-1]).squeeze(axis=-1)
    uprev = np.asarray(uprev[:-1]).squeeze(axis=-1)
    xs = np.asarray(xs).squeeze(axis=-1)
    us = np.asarray(us).squeeze(axis=-1)
    u = np.asarray(u).squeeze(axis=-1)
    # Get the time taken to generate the data.
    tend_data_gen = time.time()
    data_gen_time = tend_data_gen - tstart_data_gen
    # Return.
    return H5pyTool.save_training_data(dictionary=dict(x=x, uprev=uprev, 
                                                xs=xs, us=us, u=u,
                                                data_gen_time=data_gen_time), 
            filename=str(task_number)+'-'+str(process_number)+'-'+data_filename)