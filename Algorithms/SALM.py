import numpy as np
import copt

from Algorithms import panoc_plus
from Utilities.AugmentedLagrangianGradient import AugmentedLagrangianGradient
from results import OptimizationResult


def salm(f,
         phi,
         x0,
         g=None,
         h=None,
         f_grad=None,
         jac_g=None,
         jac_h=None,
         pen0=1,
         pen_max=1e20,
         gamma=2,
         lam0=None,
         mu0=None,
         lam_max=1e8,
         mu_max=1e8,
         tol=1e-6,
         tol_prim = None,
         tol_dual = None,
         inner_tol=None,
         kappa_tol = 0.1,
         maxiter=200,
         tau=0.8,
         subsolver = 'ProxGrad',
         ):
    """
        Implementation of the safeguarded augmented Lagrangian method (SALM) for constrained composite optimization

        Parameters
        ----------
        f : callable
            smooth part of the objective
        f_grad : callable
            gradient of smooth part
        phi : callable
            non-smooth part, has a proximal operator phi.prox
        x0 : array_like
            initial point
        g : callable or  array of callables
            inequality constraint functions
        jac_g : callable or  array of callables
            Jacobians of inequality constraint functions
        h : callable or  array of callables
            equality constraint functions
        jac_h : callable or  array of callables
            Jacobians of equality constraint functions
        pen0 : float
            initial value of penalty parameter
        pen_max : float
            upper bound for penalty parameter
        gamma : float
            penalty increase parameter > 1
        lam0 : array_like
            initial value of inequality constraint multipliers
        mu0 : array_like
            initial value of equality constraint multipliers
        lam_max :
            upper bound for inequality constraint multipliers
        mu_max :
            upper bound for equality constraint multipliers
        tol : float
            tolerance parameter for stopping criterion
        tol_prim : float
            tolerance for measuring the constraint violations
        tol_dual : float
            tolerance for measuring approximate stationarity of the Lagrangian
        inner_tol : float
            tolerance for the subproblem solver
        kappa_tol : float
            tolerance reduction parameter
        maxiter : int
            upper bound for iterations
        tau : float
            sufficient decrease parameter
        subsolver : str
            specifies which subsolver to use - default is proximal gradient

        Returns
        -------
        OptimizationResult
            optimization result of type OptimizationResult

        """
    # Initialization
    x = np.asarray(x0, dtype=float)
    if tol_prim is None:
        tol_prim = tol
    if tol_dual is None:
        tol_dual = tol
    if inner_tol is None:
        inner_tol = np.cbrt(tol)

    g = g or []
    h = h or []

    # problem dimensions are set
    n = x.size
    if callable(g):
        m = len(g(x0))
    else:
        m = len(g)
    if callable(h):
        p = len(h(x0))
    else:
        p = len(h)

    # initialize multipliers
    if lam0 is None:
        lam = np.zeros(m)
    else:
        lam = np.asarray(lam0, dtype=float)
    if mu0 is None:
        mu = np.zeros(p)
    else:
        mu = np.asarray(mu0, dtype=float)

    pen = float(pen0)

    # returns evaluation vector of the equality constraints
    def eval_h(x_value):
        if callable(h):
            return np.atleast_1d(h(x_value))
        if p < 1:
            return []
        l = [h_i(x_value) for h_i in h]
        return np.asarray(l)

    # returns evaluation vector of the inequality constraints
    def eval_g(x_value):
        if callable(g):
            return np.atleast_1d(g(x_value))
        if m < 1:
            return []
        return np.asarray([g_i(x_value) for g_i in g])


    # evaluates the augmented lagrangian function
    def augmented_lagrangian(x_value, lam_temp, mu_temp, pen_temp):
        f_val = float(f(x_value))
        g_temp = eval_g(x_value)
        h_temp = eval_h(x_value)
        eq_term = np.dot(mu_temp, h_temp) + 0.5 * pen_temp * np.dot(h_temp, h_temp) if len(h_temp) > 0 else 0.0
        g_pos = np.maximum(0.0, g_temp)
        ineq_term = np.dot(lam_temp, g_pos) + 0.5 * pen_temp * np.dot(g_pos, g_pos) if len(g_temp) > 0 else 0.0
        return f_val + ineq_term + eq_term

    # SALM Loop
    previous_ineq_norm = np.inf
    previous_eq_norm = np.inf

    # safeguarding of the initial multipliers
    u = np.clip(lam, 0.0, lam_max)
    v = np.clip(mu, -mu_max, mu_max)


    tot_it = 0
    tot_inner_it = 0
    solved = False
    tired = tot_it >= maxiter
    broken = False
    can_stop = solved or tired or broken

    while not can_stop:
        tot_it += 1
        # define objective function for the subproblem solver
        def inner_fun(x_value):
            return augmented_lagrangian(x_value, u, v, pen)

        # define augmented Lagrangian gradient for the subproblem solver
        inner_jac = AugmentedLagrangianGradient(g=g,h=h,f_grad=f_grad,jac_g=jac_g,jac_h=jac_h,lam_temp=u,mu_temp=v,pen_temp=pen,m=m,n=n,p=p)
        if phi.prox is not None:
            if subsolver == 'PANOC_plus':
                # subproblem solver PANOC+
                res = panoc_plus.panoc_plus(fun=inner_fun, grad=inner_jac, phi=phi, x0=x, verbose=False, tol=inner_tol,
                                            A=np.eye(n), maxiter=500)
                sub_solved = res.success
                # check if PANOC+ gamma got to small
                broken = res.status == 2
            else:
                # subproblem solver proximal gradient
                res = copt.minimize_proximal_gradient(fun=inner_fun, jac=inner_jac, prox=phi.prox, x0=x, verbose=2, tol=inner_tol)
                sub_solved = res.success
            tot_inner_it += res.nit
        else:
            print("No suitable prox-operator!")
            break
        x_new = res.x

        # multiplier update inequality constraints
        if m > 0:
            g_new = eval_g(x_new)
            lam_new = np.maximum(0.0, u + pen * g_new)
            ineq_norm = np.linalg.norm(np.minimum(-g_new, (1 / pen) * u))
        else:
            g_new = 0.0
            ineq_norm = 0.0
            lam_new = u

        # multiplier update equality constraints
        if p > 0:
            h_new = eval_h(x_new)
            mu_new = np.maximum(0.0, v + pen * h_new)
            eq_norm = np.linalg.norm(h_new)
        else:
            h_new = 0.0
            eq_norm = 0.0
            mu_new = v

        # Termination criteria
        eq_violation = np.linalg.norm(np.atleast_1d(h_new),ord = np.inf)
        ineq_violation = np.linalg.norm(np.minimum(-g_new,lam_new),ord = np.inf)

        solved = (inner_tol <= tol_dual and sub_solved) and eq_violation <= tol_prim and ineq_violation <= tol_prim
        tired = tot_it >= maxiter
        can_stop = solved or tired or broken

        if not can_stop:
            # Penalty update
            if ineq_norm > tau * previous_ineq_norm or eq_norm > tau * previous_eq_norm:
                pen = np.minimum(gamma * pen, pen_max)
            # Decrease inner tolerance
            inner_tol = np.maximum(kappa_tol * inner_tol,tol_dual)
        # set up the constraint norms and multipliers for the next loop
        u = np.clip(lam_new, 0.0, lam_max)
        v = np.clip(mu_new, -mu_max, mu_max)
        previous_ineq_norm = ineq_norm
        previous_eq_norm = eq_norm
        x = x_new

    print(f"lambda = {lam_new}, mu = {mu_new}")
    print(f"u = {u}, v = {v}")
    if solved:
        return OptimizationResult(x = x,
                              success = True,
                              status = 0,
                              message = f"Success! Converged in {tot_it} iterations with {tot_inner_it} inner iterations.",
                              nnz=int(np.count_nonzero(x)),
                              obj = f(x)+ phi(x),
                              outer_iterations=tot_it,
                              inner_iterations=tot_inner_it,
                              ineq_violation=float(np.linalg.norm(np.maximum(np.atleast_1d(g_new), 0.0),ord=np.inf)),
                              eq_violation=float(np.linalg.norm(np.atleast_1d(h_new),ord=np.inf)),
                              lam=lam_new,
                              mu=mu_new
        )
    elif tired:
        return OptimizationResult(x=x,
                              success=False,
                              status=1,
                              message=f"Failure! Maximum iterations reached.",
                              nnz=int(np.count_nonzero(x)),
                              obj=f(x)+ phi(x),
                              outer_iterations=tot_it,
                              inner_iterations=tot_inner_it,
                              ineq_violation=float(np.linalg.norm(np.maximum(np.atleast_1d(g_new), 0),ord=np.inf)),
                              eq_violation=float(np.linalg.norm(np.atleast_1d(h_new),ord=np.inf)),
                              lam=lam_new,
                              mu=mu_new
                              )
    elif broken:
        print(tot_it)
        return OptimizationResult(x=x,
                              success=False,
                              status=2,
                              message=f"Failure! Stepsize gamma became to small.",
                              nnz=int(np.count_nonzero(x)),
                              obj = f(x)+phi(x),
                              outer_iterations=tot_it,
                              inner_iterations=tot_inner_it,
                              ineq_violation=float(np.linalg.norm(np.maximum(np.atleast_1d(g_new), 0),ord=np.inf)),
                              eq_violation=float(np.linalg.norm(np.atleast_1d(h_new),ord=np.inf)),
                              lam=lam_new,
                              mu=mu_new
                              )
    else:
        return OptimizationResult(x=x,
                              success=False,
                              status=3,
                              message="Unknown",
                              nnz=int(np.count_nonzero(x)),
                              obj = f(x)+phi(x),
                              outer_iterations=tot_it,
                              inner_iterations=tot_inner_it,
                              ineq_violation=float(np.linalg.norm(np.maximum(np.atleast_1d(g_new), 0),ord=np.inf)),
                              eq_violation=float(np.linalg.norm(np.atleast_1d(h_new),ord=np.inf)),
                              lam=lam_new,
                              mu=mu_new
                              )
