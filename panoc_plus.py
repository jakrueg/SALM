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
import LBFGS


# ============================================================
# Result object (SciPy-like)
# ============================================================
@dataclass
class OptimizeResult:
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
    def __init__(self, x0, A, gamma, lbfgs, f, grad_f, prox_g):
        self.x = x0.copy()
        self.A = A
        self.gamma = gamma
        self.lbfgs = lbfgs
        self.f = f
        self.grad_f = grad_f
        self.prox_g = prox_g

        self.x_prev = self.x.copy()
        self.res_prev = np.zeros_like(self.x)
        self.d = np.zeros_like(self.x)

        self._update_forward_backward()

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
# PANOC+ solver (SciPy-like)
# ============================================================
def panoc_plus(
    fun,
    grad,
    prox,
    x0,
    A=None,
    gamma=None,
    maxiter=1000,
    tol=1e-8,
    alpha=0.95,
    beta=0.5,
    lbfgs_memory=5,
    adaptive=True,
    min_gamma=1e-7,
    max_backtracks=20,
    verbose=False,
):
    """
    Minimize f(Ax) + g(x) using PANOC+.

    Parameters (similar to scipy.optimize.minimize):
    ------------------------------------------------
    fun(x):          f(x)  (smooth part)
    grad(x):         ∇f(x)
    prox(y, gamma):  prox_{gamma g}(y) -> (z, g(z))
    x0:              initial point
    A:               linear operator (default: identity)
    gamma:           stepsize (if None, must be adapted externally)
    """

    if A is None:
        A = np.eye(len(x0))

    #if gamma is None:
    #    raise ValueError("PANOC+ requires an initial gamma")

    lbfgs = LBFGS.LBFGSOperator(lbfgs_memory,x0)
    state = _PANOCState(x0, A, gamma, lbfgs, fun, grad, prox)

    for k in range(1, maxiter + 1):
        state.x_prev[:] = state.x
        state.res_prev[:] = state.res

        FBE_x = _fbe_model(state, alpha) + state.g_z
        sigma = beta * (0.5 / state.gamma) * (1 - alpha)
        threshold = FBE_x - sigma * np.linalg.norm(state.res) ** 2

        tau = 1.0
        can_update_direction = True
        backtracks = 0

        while True:
            if can_update_direction:
                state.d = -1.0 * (lbfgs @ state.res_prev)
                state.x = state.x_prev + state.d
                tau = 1.0
                backtracks = 0
            else:
                state.x = (
                    (1 - tau) * (state.x_prev - state.res_prev)
                    + tau * (state.x_prev + state.d)
                )
                backtracks += 1

            state._update_forward_backward()
            fbe_new = _fbe_model(state, alpha) + state.g_z

            if adaptive:
                if fbe_new > FBE_x and state.gamma >= min_gamma:
                    state.gamma *= 0.5
                    lbfgs.reset()
                    can_update_direction = True
                    continue

            if fbe_new <= threshold or backtracks >= max_backtracks:
                break

            tau = 0.0 if backtracks >= max_backtracks - 1 else tau / 2.0
            can_update_direction = False

        # L-BFGS update
        lbfgs.update(state.x - state.x_prev, state.res - state.res_prev)

        # stopping criterion (Bazinga.jl default)
        crit = np.linalg.norm(
            (state.res / state.gamma)
            - state.At_grad_f_Ax
            + state.At_grad_f_Az,
            ord=np.inf,
        )
        if verbose:
            print(f"{k:4d} | res={crit:.2e} | gamma={state.gamma:.2e}")

        if crit <= tol:
            return OptimizeResult(
                x=state.z,
                success=True,
                status=0,
                message="Converged",
                nit=k,
                fun=fun(A @ state.z) + state.g_z,
                residual_norm=crit,
                gamma=state.gamma,
            )

    return OptimizeResult(
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
