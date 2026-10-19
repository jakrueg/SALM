import numpy as np
from dataclasses import dataclass


@dataclass
class OptimizeResultProxGrad:
    x: np.ndarray
    success: bool
    status: int
    message: str
    nit: int
    fun: float
    residual_norm: float
    gamma: float

def proximal_gradient(
    fun, grad, phi, x0,
    tau0=1.0,
    beta=0.5,
    max_iter=10_000,
    tol=1e-6,
):
    """
    Minimize fun(x) + phi(x).

    fun(x): smooth objective F(x)
    grad(x): gradient of F(x)
    prox(z, tau): global minimizer of
                  phi(y) + ||y-z||^2 / (2*tau)
    """
    x = np.asarray(x0, dtype=float).copy()
    tau = tau0
    delta = 0.9
    MAX_BACKTRACK = 30

    for k in range(max_iter):
        fx = fun(x)
        phix = phi(x)
        gx = grad(x)

        # Backtracking on the smooth part
        accepted = False
        for _ in range(MAX_BACKTRACK):
            tau_try = tau
            x_new = phi.prox(x - tau_try * gx, tau_try)
            d = x_new - x

            upper = fx + phix - delta * (d @ d) / (2 * tau_try)

            if fun(x_new) + phi(x_new) <= upper:
                accepted = True
                break

            tau *= beta

        if not accepted:
            #print("Backtracking failed to find an acceptable step.")
            return OptimizeResultProxGrad(
                x=x,
                success=False,
                status=2,
                message="Backtracking failed to find an acceptable step.",
                nit=k,
                fun=fun(x),
                residual_norm=float("inf"),
                gamma=tau_try,
            )
        tau = tau_try
        # Proximal-gradient mapping residual
        g_new = grad(x_new)
        residual = np.linalg.norm(g_new-gx-d/tau) 

        x = x_new

        if residual <= tol:
            return  OptimizeResultProxGrad(
                x=x,
                success=True,
                status=0,
                message="Optimization terminated successfully.",
                nit=k + 1,
                fun=fun(x),
                residual_norm=residual,
                gamma=tau,
            )

    return  OptimizeResultProxGrad(
        x=x,
        success=False,
        status=1,
        message="Optimization failed to converge.",
        nit=max_iter,
        fun=fun(x),
        residual_norm=residual,
        gamma=tau,
    )       