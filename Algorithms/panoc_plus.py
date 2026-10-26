# ============================================================
# PANOC+ solver with L-BFGS inverse Jacobian
#
# Reference:
# https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/blob/master/src/algorithms/panocplus.jl
# ============================================================

import numpy as np
from dataclasses import dataclass
from Utilities import LBFGS

# result object

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


# PANOC+ internal state
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


# Forward–Backward Envelope

def _fbe_model(state, alpha):
    return (
        state.f_Ax
        - np.dot(state.At_grad_f_Ax, state.res)
        + (alpha / (2.0 * state.gamma)) * np.linalg.norm(state.res) ** 2
    )

# auxiliary functions

def f_model(f_x,grad_f_x,res,L):
    return f_x- np.dot(grad_f_x, res)+ 0.5*L*np.linalg.norm(res)**2

def lower_bound_smoothness_constant_inner(f,grad_f,A,x,grad_f_Ax):
    xeps = x + 1
    grad_f_Axeps = grad_f(A @ xeps)
    return np.linalg.norm(A.T @ (grad_f_Axeps-grad_f_Ax))/np.sqrt(len(x))

def lower_bound_smoothness_constant(f,grad_f,A,x):
    Ax = A @ x
    grad_f_Ax = grad_f(Ax)
    return lower_bound_smoothness_constant_inner(f,grad_f,A,x, grad_f_Ax)




def panoc_plus(
    fun,
    phi,
    grad,
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
    max_backtracks=20,
    verbose=False,
):
    """
    Implementation of the PANOC+ algorithm by De Marchi, Themelis

    Parameters
    ----------
    fun : callable
        smooth part of the objective
    grad : callable
        gradient of smooth part
    phi : callable
        non-smooth part, has a proximal operator phi.prox
    x0 : array_like
        initial point
    A  :
        linear operator (default: identity)
    gamma : float
        stepsize
    maxiter : int
        upper limit for PANOC+ iterations
    tol : float
        tolerance for stopping criterion
    alpha : float
        first sufficient decrease parameter
    beta : float
        second sufficient decrease parameter
    lbfgs_memory : int
        memory length for the limited memory BFGS method
    adaptive : bool
        flag for enabling adaptive choice of stepsize gamma
    min_gamma : float
        lower bound for stepsize
    max_backtracks : int
        upper bound for number of backtracks of tau
    verbose : bool
        verbosity option

    Returns
    -------
    OptimizeResultPanoc
        optimization result of type OptimizeResultPanoc


    """

    if A is None:
        A = np.eye(len(x0))

    x = np.copy(x0)
    Ax = A @ x
    f_Ax = fun(Ax)
    grad_f_Ax = grad(Ax)

    if gamma is None:
        # estimate smoothness constant
        L = lower_bound_smoothness_constant(f=fun, grad_f=grad, A=A, x=x0)
        if L == 0:
            gamma  = alpha
        else:
            gamma = alpha / L

    # set up initial algorithm state
    At_grad_f_Ax = A.T @ grad_f_Ax
    y = x - gamma * At_grad_f_Ax
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
    # first algorithm step, has to be executed separately
    if gamma is None or adaptive is True:
        reduce_gamma = 0.5
        try:
            f_Az_upp = f_model(state.f_Ax, state.At_grad_f_Ax, state.res, alpha / state.gamma)
            state.Az = A @ state.z
            f_Az = fun(state.Az)
            grad_f_Az_tmp = grad(state.Az)
            machine_tol = 10 * np.finfo(float).eps * (1 + np.abs(f_Az))
            num_iters = 0
            # gamma backtracking
            while f_Az > f_Az_upp + machine_tol and state.gamma >= min_gamma:
                state.gamma *= reduce_gamma
                state.y = state.x - state.gamma * state.At_grad_f_Ax
                state.z = phi.prox(state.y, state.gamma)
                state.g_z = phi(state.z)
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
                print(f"Initial backtrack failed after {num_iters} steps!")
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

    # PANOC+ iteration

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
                # compute direction by L-BFGS
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
            state.z = phi.prox(state.y,state.gamma)
            state.g_z = phi(state.z)
            state.res = state.x-state.z

            f_Az_upp = f_model(state.f_Ax,state.At_grad_f_Ax,state.res,alpha/state.gamma)

            state.Az = A @ state.z
            f_Az = fun(state.Az)
            grad_f_Az = grad(state.Az)
            state.grad_f_Az = grad_f_Az

            if adaptive or gamma is None:
                # backtrack gamma
                machine_tol = 10 * np.finfo(float).eps * (1 + np.abs(f_Az))
                if f_Az > f_Az_upp + machine_tol and state.gamma >= min_gamma:
                    state.gamma *= 0.5
                    if state.gamma < min_gamma:
                        print(f"Gamma backtrack failed during PANOC+ iteration after {num_iters} steps!")
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
                    state.H.reset()
                    can_update_direction = True
                    continue
            # backtrack tau
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

        # stopping criterion
        crit = float(np.linalg.norm(
            (state.res / state.gamma)
            - state.At_grad_f_Ax
            + state.At_grad_f_Az,
            ord=np.inf,
        ))
        if verbose:
            print(f"{k:4d} | res={np.linalg.norm(state.res,ord=np.inf)/state.gamma:.3e} | criterion={crit:.3e} | gamma={state.gamma:.3e} | tau={state.tau:.3e} | nnz={np.count_nonzero(state.z):4d} |")

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
