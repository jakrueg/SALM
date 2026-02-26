# ============================================================
# panoc_plus.py
#
# PANOC+ solver with L-BFGS inverse Jacobian
# SciPy-like interface (similar to scipy.optimize.minimize)
#
# Reference:
# De Marchi, Themelis (2022)
# ============================================================

import numpy as np
from dataclasses import dataclass
from Utilities import LBFGS


# ============================================================
# Result object (SciPy-like)
# ============================================================
@dataclass
class OptimizeResultPanoc:
    x: np.ndarray
    success: bool
    status: int
    message: str
    nit: int
    fun: float
    residual_norm: float
    gamma: float

"""
# ============================================================
# L-BFGS inverse Jacobian (for residuals)
# ============================================================
class LBFGSInverseJacobian:
    def __init__(self, dim, memory=5):
        self.m = memory
        self.S = []
        self.Y = []
        self.rho = []
        self.gamma = 1.0

    def reset(self):
        self.S.clear()
        self.Y.clear()
        self.rho.clear()
        self.gamma = 1.0

    def update(self, s, y):
        ys = np.dot(y, s)
        if ys <= 1e-12:
            return
        if len(self.S) == self.m:
            self.S.pop(0)
            self.Y.pop(0)
            self.rho.pop(0)
        self.S.append(s.copy())
        self.Y.append(y.copy())
        self.rho.append(1.0 / ys)
        yy = np.dot(y, y)
        if yy > 0:
            self.gamma = ys / yy

    def apply(self, q):
        q = q.copy()
        alpha = []
        for s, y, rho in zip(reversed(self.S), reversed(self.Y), reversed(self.rho)):
            a = rho * np.dot(s, q)
            alpha.append(a)
            q -= a * y
        r = self.gamma * q
        for s, y, rho, a in zip(self.S, self.Y, self.rho, reversed(alpha)):
            b = rho * np.dot(y, r)
            r += s * (a - b)
        return r
"""

# ============================================================
# PANOC+ internal state
# ============================================================
class _PANOCState:
    def __init__(self,
                 x0,
                 Ax,
                 f_Ax,
                 grad_f_Ax,
                 At_grad_f_Ax,
                 gamma,
                 #lbfgs,
                 y,
                 z,
                 g_z,
                 res,
                 H):
        self.x = x0
        self.Ax = Ax
        self.f_Ax = f_Ax
        self.grad_f_Ax = grad_f_Ax
        self.At_grad_f_Ax = At_grad_f_Ax
        self.gamma = gamma
        #self.lbfgs = lbfgs
        self.y = y
        self.z = z
        self.g_z = g_z
        self.res = res
        self.H = H

        self.tau = 0.0
        self.x_prev = np.zeros_like(self.x)
        self.res_prev = np.zeros_like(self.x)
        self.d = np.zeros_like(self.x)
        self.Az = np.zeros_like(self.Ax)
        self.grad_f_Az = np.zeros_like(self.Ax)
        self.At_grad_f_Az = np.zeros_like(self.x)
"""
    def _update_forward_backward(self):
        Ax = self.A @ self.x
        self.f_Ax = self.f(Ax)
        grad_f_Ax = self.grad_f(Ax)
        self.At_grad_f_Ax = self.A.T @ grad_f_Ax
        y = self.x - self.gamma * self.At_grad_f_Ax
        self.z, self.g_z = self.prox_g(y, self.gamma)
        self.res = self.x - self.z

        Az = self.A @ self.z
        self.grad_f_Az = self.grad_f(Az)
        self.At_grad_f_Az = self.A.T @ self.grad_f_Az
        self.Az = Az
"""

# ============================================================
# Forward–Backward Envelope model
# ============================================================
def _fbe_model(state, alpha):
    return (
        state.f_Ax
        - np.dot(state.At_grad_f_Ax, state.res)
        + (alpha / (2.0 * state.gamma)) * np.linalg.norm(state.res) ** 2
    )

# ============================================================
# Some helper functions for the initialization
# ============================================================

def f_model(f_x,grad_f_x,res,L):
    return f_x- np.dot(grad_f_x, res)+ 0.5*L*np.linalg.norm(res)**2

def lower_bound_smoothness_constant_inner(f,grad_f,A,x,grad_f_Ax):
    xeps = x + 1
    #f_Axeps = f(A @ xeps)
    grad_f_Axeps = grad_f(A @ xeps)
    #print(f"grad_f_Axeps = {grad_f_Axeps}")
    #print(f"grad_f_Ax = {grad_f_Ax}")
    #print(A @ xeps)
    return np.linalg.norm(A.T @ (grad_f_Axeps-grad_f_Ax))/np.sqrt(len(x))

def lower_bound_smoothness_constant(f,grad_f,A,x):
    Ax = A @ x
    grad_f_Ax = grad_f(Ax)
    return lower_bound_smoothness_constant_inner(f,grad_f,A,x, grad_f_Ax)

def backtrack_stepsize_inner(gamma,
                       f,
                       grad_f,
                       A,
                       phi,
                       x,
                       f_Ax,
                       At_grad_f_Ax,
                       y,
                       z,
                       g_z,
                       res,
                       Az,
                       grad_f_Az = None,
                       alpha = 1.0,
                       minimum_gamma = 1e-7,
                       reduce_gamma = 0.5
):
    f_Az_upp = f_model(f_Ax,At_grad_f_Ax,res,alpha/gamma)
    Az = A @ z
    f_Az = f(Az)
    grad_f_Az_tmp = grad_f(Az)
    machine_tol = 10*np.finfo(float).eps * (1+np.abs(f_Az))
    #print(f"tol = {tol}")
    #print(f"Initial backtracking: f(Az) = {f_Az}, upper bound = {f_Az_upp}, gamma = {gamma}")
    num_iters = 0
    while f_Az > f_Az_upp + machine_tol and gamma >=minimum_gamma:
        gamma *= reduce_gamma
        y = x -gamma*At_grad_f_Ax
        z = phi.prox(y,gamma)
        g_z = phi(z)
        #z, g_z = phi_prox(y,gamma)
        res = x-z
        f_Az_upp = f_model(f_Ax,At_grad_f_Ax,res,alpha/gamma)
        Az = A @ z
        f_Az = f(Az)
        grad_f_Az_tmp = grad_f(Az)
        machine_tol = 10*np.finfo(float).eps * (1+np.abs(f_Az))
        num_iters += 1
    if not grad_f_Az is None:
        grad_f_Az = grad_f_Az_tmp
    if gamma < minimum_gamma:
        #print(f"num_iters = {num_iters}")
        raise ValueError("Stepsize gamma became to small!")
    return gamma, g_z,f_Az,f_Az_upp

def backtrack_stepsize(gamma,
                       f,
                       grad_f,
                       A,
                       phi,
                       x,
                       alpha = 1.0,
                       minimum_gamma = 1e-7,
                       reduce_gamma = 0.5
                       ):
    Ax = A @ x
    f_Ax = f(Ax)
    grad_f_Ax = grad_f(Ax)
    At_grad_f_Ax = A.T @ grad_f_Ax
    y = x -gamma*At_grad_f_Ax
    #z, g_z = phi_prox(y,gamma)
    z = phi.prox(y,gamma)
    g_z = phi(z)
    return backtrack_stepsize_inner(gamma=gamma,
                              f=f,
                              grad_f=grad_f,
                              A=A,
                              phi=phi,
                              x=x,
                              f_Ax=f_Ax,
                              At_grad_f_Ax=At_grad_f_Ax,
                              y=y,
                              z=z,
                              g_z=g_z,
                              res=x-z,
                              Az=Ax,
                              grad_f_Az=grad_f_Ax,
                              alpha=alpha,
                              minimum_gamma=minimum_gamma,
                              reduce_gamma=reduce_gamma)
# ============================================================
# Helper functions for the iterations
# ============================================================



# ============================================================
# PANOC+ solver (SciPy-like)
# ============================================================
def panoc_plus(
    fun,
    phi,
    grad,
    #prox,
    x0,
    A=None,
    gamma=None,
    maxiter=1000,
    tol=1e-6,
    alpha=0.95,
    beta=0.5,
    lbfgs_memory=5,
    adaptive=True,
    min_gamma=1e-32,
    max_backtracks=50,
    verbose=False,
):
    """
    Minimize f(Ax) + g(x) using PANOC+.

    Parameters (similar to scipy.optimize.minimize):
    ------------------------------------------------
    fun(x):          f(x)  (smooth part)
    jac(x):         ∇f(x)
    phi(x):         phi(x) (non-smooth part, has a proximal operator phi.prox)
    x0:              initial point
    A:               linear operator (default: identity)
    gamma:           stepsize (if None, must be adapted externally)
    """

    if A is None:
        A = np.eye(len(x0))

    x = np.copy(x0)
    Ax = A @ x
    f_Ax = fun(Ax)
    grad_f_Ax = grad(Ax)

    if gamma is None:
        gamma = alpha / lower_bound_smoothness_constant(f=fun, grad_f=grad, A=A, x=x0)
        #gamma = alpha* 1e-2
        #gamma = 1.0
        #print(f"gamma = {gamma}")

    At_grad_f_Ax = A.T @ grad_f_Ax
    y = x - gamma * At_grad_f_Ax
    #z, g_z = phi_prox(y,gamma)
    z = phi.prox(y,gamma)
    g_z = phi(z)

    lbfgs = LBFGS.LBFGSOperator(lbfgs_memory, x)
    state = _PANOCState(x0 = x,
                        Ax=Ax,
                        f_Ax = f_Ax,
                        grad_f_Ax = grad_f_Ax,
                        At_grad_f_Ax = At_grad_f_Ax,
                        gamma = gamma,
                        y = y,
                        z = z,
                        g_z = g_z,
                        res = x-z,
                        H = lbfgs
                        )

    if gamma is None or adaptive is True:
        reduce_gamma = 0.5
        try:
            """
            state.gamma, state.g_z , _ , _ = backtrack_stepsize_inner(
                gamma = state.gamma,
                f= fun,
                grad_f= grad,
                A=A,
                phi=phi,
                x= state.x,
                f_Ax= state.f_Ax,
                At_grad_f_Ax= state.At_grad_f_Ax,
                y= state.y,
                z =state.z,
                g_z =state.g_z,
                res =state.res,
                Az =state.Az,
                grad_f_Az= state.grad_f_Az,
                alpha = alpha,
                minimum_gamma = min_gamma
                
            )
            """
            f_Az_upp = f_model(state.f_Ax, state.At_grad_f_Ax, state.res, alpha / state.gamma)
            state.Az = A @ state.z
            f_Az = fun(state.Az)
            grad_f_Az_tmp = grad(state.Az)
            machine_tol = 10 * np.finfo(float).eps * (1 + np.abs(f_Az))
            # print(f"tol = {tol}")
            #print(f"Initial backtracking: f(Az) = {f_Az}, upper bound = {f_Az_upp}, gamma = {gamma}")
            num_iters = 0
            while f_Az > f_Az_upp + machine_tol and state.gamma >= min_gamma:
                state.gamma *= reduce_gamma
                state.y = state.x - state.gamma * state.At_grad_f_Ax
                state.z = phi.prox(state.y, state.gamma)
                state.g_z = phi(state.z)
                # z, g_z = phi_prox(y,gamma)
                state.res = state.x - state.z
                f_Az_upp = f_model(state.f_Ax, state.At_grad_f_Ax, state.res, alpha / state.gamma)
                state.Az = A @ state.z
                f_Az = fun(state.Az)
                grad_f_Az_tmp = grad(state.Az)
                machine_tol = 10 * np.finfo(float).eps * (1 + np.abs(f_Az))
                num_iters += 1
            if not state.grad_f_Az is None:
                state.grad_f_Az = grad_f_Az_tmp
            if state.gamma < min_gamma:
                print(f"Initial backtracking: f(Az) = {f_Az}, upper bound = {f_Az_upp}, gamma = {state.gamma}")
                print(f"Initial backtrack! num_iters = {num_iters}")
                raise ValueError("Stepsize gamma became to small!")
        except ValueError:
            return OptimizeResultPanoc(
                x=state.z,
                success=False,
                status=2,
                message="Stepsize gamma became to small!",
                nit=maxiter,
                fun=fun(A @ state.z) + state.g_z,
                residual_norm=float(np.linalg.norm(
                    (state.res / state.gamma)
                    - state.At_grad_f_Ax
                    + state.At_grad_f_Az,
                    ord=np.inf,
                )),
                gamma=state.gamma,
            )

    else:
        state.Az = A @ state.z
        grad_f_Az = grad(state.z)
        state.grad_f_Az = grad_f_Az

    state.At_grad_f_Az = A.T @ state.grad_f_Az

    for k in range(1, maxiter + 1):
        # store iterate and residual for metric update later on
        state.x_prev[:] = state.x
        state.res_prev[:] = state.res

        # compute FBE
        FBE_x = _fbe_model(state, alpha) + state.g_z

        sigma = beta * (0.5 / state.gamma) * (1 - alpha)
        machine_tol = 10*np.finfo(float).eps * (1+np.abs(FBE_x))
        threshold = FBE_x - sigma * np.linalg.norm(state.res) ** 2+machine_tol

        can_update_direction = True
        backtracks = 0

        while True:
            if can_update_direction:
                # compute direction
                state.d = -1.0 * (state.H @ state.res_prev)
                state.tau = 1.0
                state.x = state.x_prev + state.d
                backtracks = 0
            else:
                state.x = (
                    (1 - state.tau) * (state.x_prev - state.res_prev)
                    + state.tau * (state.x_prev + state.d)
                )
                backtracks += 1

            state.Ax = A @ state.x
            state.f_Ax = fun(state.Ax)
            state.grad_f_Ax = grad(state.Ax)
            state.At_grad_f_Ax = A.T @ state.grad_f_Ax

            state.y = state.x -state.gamma * state.At_grad_f_Ax
            #state.z, state.g_z = prox(state.y,state.gamma)
            state.z = phi.prox(state.y,state.gamma)
            state.g_z = phi(state.z)
            state.res = state.x-state.z

            f_Az_upp = f_model(state.f_Ax,state.At_grad_f_Ax,state.res,alpha/state.gamma)

            state.Az = A @ state.z
            f_Az = fun(state.Az)
            grad_f_Az = grad(state.Az)
            state.grad_f_Az = grad_f_Az

            if adaptive or gamma is None:
                machine_tol = 10 * np.finfo(float).eps * (1 + np.abs(f_Az))
                if f_Az > f_Az_upp + machine_tol and state.gamma >= min_gamma:
                    state.gamma *= 0.5
                    if state.gamma < min_gamma:
                        print("In iteration Process")
                        return OptimizeResultPanoc(
                            x=state.z,
                            success=False,
                            status=2,
                            message="Stepsize gamma became to small!",
                            nit=maxiter,
                            fun=fun(A @ state.z) + state.g_z,
                            residual_norm=float(np.linalg.norm(
                                (state.res / state.gamma)
                                - state.At_grad_f_Ax
                                + state.At_grad_f_Az,
                                ord=np.inf,
                                )),
                            gamma=state.gamma,
                        )
                        #print(k)
                        #print(f_Az)
                        #print(f_Az_upp)
                        raise ValueError("Stepsize gamma became to small!")
                    state.H.reset()
                    can_update_direction = True
                    continue
            state.At_grad_f_Az = A.T @ state.grad_f_Az

            FBE_x_new = f_Az_upp + state.g_z
            if FBE_x_new <= threshold or backtracks >= max_backtracks:
                break
            state.tau = 0.0 if backtracks >= max_backtracks - 1 else state.tau / 2.0
            can_update_direction = False

        # L-BFGS update
        state.x_prev = state.x -state.x_prev
        state.res_prev = state.res-state.res_prev
        state.H.update(state.x_prev, state.res_prev)

        # stopping criterion (Bazinga.jl default)
        crit = float(np.linalg.norm(
            (state.res / state.gamma)
            - state.At_grad_f_Ax
            + state.At_grad_f_Az,
            ord=np.inf,
        ))
        if verbose:
            print(f"{k:4d} | res={np.linalg.norm(state.res,ord=np.inf)/state.gamma:.3e} | criterion={crit:.3e} | gamma={state.gamma:.3e} | tau={state.tau:.3e}")

        if crit <= tol:
            return OptimizeResultPanoc(
                x=state.z,
                success=True,
                status=0,
                message="Converged",
                nit=k,
                fun=fun(A @ state.z) + state.g_z,
                residual_norm=crit,
                gamma=state.gamma,
            )

    return OptimizeResultPanoc(
        x=state.z,
        success=False,
        status=1,
        message="Maximum iterations reached",
        nit=maxiter,
        fun=fun(A @ state.z) + state.g_z,
        residual_norm=crit,
        gamma=state.gamma,
    )


# ============================================================
# End of file
# ============================================================
