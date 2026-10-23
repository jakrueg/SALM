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
    initial_gradient_norm: float = float("nan")
    gradient_norm: float = float("nan")
    step_norm: float = float("nan")
    backtracking_steps: int = 0
    max_backtracking_steps: int = 0
    initial_gamma: float = float("nan")
    min_gamma: float = float("nan")
    finite: bool = True

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
    initial_gamma = tau
    min_gamma = tau
    total_backtracking_steps = 0
    max_backtracking_steps = 0
    delta = 0.9
    MAX_BACKTRACK = 30

    for k in range(max_iter):
        fx = fun(x)
        phix = phi(x)
        gx = grad(x)
        if k == 0:
            initial_gradient_norm = float(np.linalg.norm(gx, ord=np.inf))

        # Backtracking on the smooth part
        accepted = False
        backtracking_steps = 0
        for _ in range(MAX_BACKTRACK):
            tau_try = tau
            x_new = phi.prox(x - tau_try * gx, tau_try)
            d = x_new - x

            upper = fx + phix - delta * (d @ d) / (2 * tau_try)

            if fun(x_new) + phi(x_new) <= upper:
                accepted = True
                break

            tau *= beta
            backtracking_steps += 1

        min_gamma = min(min_gamma, tau_try)
        if not accepted:
            total_backtracking_steps += backtracking_steps
            max_backtracking_steps = max(max_backtracking_steps, backtracking_steps)
            #print(f"Backtracking failed to find an acceptable step at iteration {k}.")
            return OptimizeResultProxGrad(
                x=x,
                success=False,
                status=2,
                message="Backtracking failed to find an acceptable step.",
                nit=k,
                fun=fun(x),
                residual_norm=float("inf"),
                gamma=tau,
                initial_gradient_norm=initial_gradient_norm,
                gradient_norm=float(np.linalg.norm(gx, ord=np.inf)),
                step_norm=0.0,
                backtracking_steps=total_backtracking_steps,
                max_backtracking_steps=max_backtracking_steps,
                initial_gamma=initial_gamma,
                min_gamma=min_gamma,
                finite=bool(np.all(np.isfinite(x)) and np.all(np.isfinite(gx))),
            )
        tau = tau_try
        total_backtracking_steps += backtracking_steps
        max_backtracking_steps = max(max_backtracking_steps, backtracking_steps)
        g_new = grad(x_new)
        residual = np.linalg.norm(g_new-gx-d/tau,np.inf) 

        x = x_new
        finite = bool(
            np.all(np.isfinite(x))
            and np.all(np.isfinite(g_new))
            and np.isfinite(residual)
            and np.isfinite(fun(x))
        )

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
                initial_gradient_norm=initial_gradient_norm,
                gradient_norm=float(np.linalg.norm(g_new, ord=np.inf)),
                step_norm=d,
                backtracking_steps=total_backtracking_steps,
                max_backtracking_steps=max_backtracking_steps,
                initial_gamma=initial_gamma,
                min_gamma=min_gamma,
                finite=finite,
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
        initial_gradient_norm=initial_gradient_norm,
        gradient_norm=float(np.linalg.norm(grad(x), ord=np.inf)),
        step_norm=d,
        backtracking_steps=total_backtracking_steps,
        max_backtracking_steps=max_backtracking_steps,
        initial_gamma=initial_gamma,
        min_gamma=min_gamma,
        finite=bool(np.all(np.isfinite(x)) and np.isfinite(residual)),
    )