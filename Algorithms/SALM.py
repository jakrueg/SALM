import numpy as np
import copt
from scipy.sparse import lil_matrix


from Algorithms import panoc_plus
from Utilities.AugmentedLagrangianGradient import AugmentedLagrangianGradient
from results import OptimizationResult


def salm(f,
         phi,
         x0,
         g=None,
         h=None,
         f_grad=None,
         #phi_prox=None,
         jac_g=None,
         jac_h=None,
         pen0=1,
         pen_max=1e8,
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
    # Initialization
    x = np.asarray(x0, dtype=float)
    if tol_prim is None:
        tol_prim = tol
    if tol_dual is None:
        tol_dual = tol
    if inner_tol is None:
        inner_tol = np.cbrt(tol)
    #print(f(x))
    #print(tol)
    g = g or []
    h = h or []
    n = x.size
    if callable(g):
        m = len(g(x0))
    else:
        m = len(g)
    if callable(h):
        p = len(h(x0))
    else:
        p = len(h)

    if not callable(jac_g):
        J_g = lil_matrix((m, n), dtype=float)

    if not callable(jac_h):
        J_h = lil_matrix((p, n), dtype=float)
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

    # returns evaluation matrix for the jacobian of h
    def eval_jac_h(x_value):
        if not jac_h:
            return None
        if callable(jac_h):
            return np.atleast_1d(jac_h(x_value))
        eval_jac_into(J_h,x_value,jac_h)
        return J_h.tocsr()
        #return np.vstack([np.atleast_1d(jac_h_i(x_value)) for jac_h_i in jac_h])

    # Auxiliary function that works on preallocated matrix

    def eval_jac_into(J, x_value,jac_func):
        J.rows = [[] for _ in range(J.shape[0])]
        J.data = [[] for _ in range(J.shape[0])]

        for i, jac_i in enumerate(jac_func):
            row = np.atleast_1d(jac_i(x_value))
            nz = np.nonzero(row)[0]
            J.rows[i] = nz.tolist()
            J.data[i] = row[nz].tolist()

    # returns evaluation matrix for the jacobian of g
    def eval_jac_g(x_value):
        if not jac_g:
            return None
        if callable(jac_g):
            return np.atleast_1d(jac_g(x_value))
        eval_jac_into(J_g,x_value,jac_g)
        return J_g.tocsr()
        #return np.vstack([np.atleast_1d(jac_g_i(x_value)) for jac_g_i in jac_g])



    # evaluates the augmented lagrangian function
    def augmented_lagrangian(x_value, lam_temp, mu_temp, pen_temp):
        #f_val = float(f(x_value))+float(phi(x_value))
        f_val = float(f(x_value))
        g_temp = eval_g(x_value)
        h_temp = eval_h(x_value)
        eq_term = np.dot(mu_temp, h_temp) + 0.5 * pen_temp * np.dot(h_temp, h_temp) if len(h_temp) > 0 else 0.0
        g_pos = np.maximum(0.0, g_temp)
        ineq_term = np.dot(lam_temp, g_pos) + 0.5 * pen_temp * np.dot(g_pos, g_pos) if len(g_temp) > 0 else 0.0
        return f_val + ineq_term + eq_term

    # computes the gradient of the differentiable part of the Augmented Lagrangian
    def augmented_lagrangian_grad(x_value, lam_temp, mu_temp, pen_temp):
        if f_grad is None:
            return None
        f_grad_val = np.asarray(f_grad(x_value), dtype=float)
        grad_sum = f_grad_val.copy()
        h_temp = eval_h(x_value)
        #print(h_temp)
        h_Jac_temp = eval_jac_h(x_value)
        if len(h_temp) > 0:
            grad_sum += np.transpose(np.matmul(np.transpose(h_Jac_temp), mu_temp + pen_temp * h_temp))
        g_temp = eval_g(x_value)
        g_Jac_temp = eval_jac_g(x_value)
        #print(g_Jac_temp)
        if len(g_temp) > 0:
            #grad_sum += g_Jac_temp.T.dot(lam_temp+pen_temp*np.maximum(0.0,g_temp))
            grad_sum += np.transpose(np.matmul(np.transpose(g_Jac_temp), lam_temp + pen_temp * np.maximum(0.0, g_temp)))
        #print(grad_sum)
        return grad_sum

    # SALM Loop

    previous_ineq_norm = np.inf
    previous_eq_norm = np.inf
    u = np.clip(lam, 0.0, lam_max)
    v = np.clip(mu, -mu_max, mu_max)
    augLagGrad = AugmentedLagrangianGradient(g=g,h=h,f_grad=f_grad,jac_g=jac_g,jac_h=jac_h,lam_temp=u,mu_temp=v,pen_temp=pen,m=m,n=n,p=p)
    #previous_aug_lag_grad = augmented_lagrangian_grad(x0, u, v, pen)
    previous_aug_lag_grad = augLagGrad(x0)

    tot_it = 0
    tot_inner_it = 0
    solved = False
    tired = tot_it >= maxiter
    broken = False
    can_stop = solved or tired or broken
    #for k in range(maxiter):
    while not can_stop:
        tot_it += 1
        # define inner function for solution
        def inner_fun(x_value):
            return augmented_lagrangian(x_value, u, v, pen)

        #def inner_jac(x_value):
        #    return augmented_lagrangian_grad(x_value, u, v, pen)
        inner_jac = AugmentedLagrangianGradient(g=g,h=h,f_grad=f_grad,jac_g=jac_g,jac_h=jac_h,lam_temp=u,mu_temp=v,pen_temp=pen,m=m,n=n,p=p)
        sub_solved = False
        if phi.prox is not None:
            if subsolver == 'PANOC_plus':
                res = panoc_plus.panoc_plus(fun=inner_fun, grad=inner_jac, phi=phi, x0=x, verbose=False, tol=inner_tol,
                                            A=np.eye(n), maxiter=200)
                sub_solved = res.success
                broken = res.status == 2
            else:
                res = copt.minimize_proximal_gradient(fun=inner_fun, jac=inner_jac, prox=phi.prox, x0=x, verbose=2, tol=inner_tol)
                sub_solved = res.success
            #print(res.x)
            #print(res.nit)
            #print(res.success)
            tot_inner_it += res.nit
        else:
            print("No suitable prox-operator!")
            break
        x_new = res.x
        if m > 0:
            g_new = eval_g(x_new)
            lam_new = np.maximum(0.0, u + pen * g_new)
            ineq_norm = np.linalg.norm(np.minimum(-g_new, (1 / pen) * u))
        else:
            g_new = 0.0
            ineq_norm = 0.0
            lam_new = u

        if p > 0:
            h_new = eval_h(x_new)
            mu_new = np.maximum(0.0, v + pen * h_new)
            eq_norm = np.linalg.norm(h_new)
        else:
            h_new = 0.0
            eq_norm = 0.0
            mu_new = v

        # Termination criteria



        # grad_norm = np.linalg.norm(augmented_lagrangian_grad(x_new, lam_new, mu_new, pen))
        scale = 1.0
        augLagGrad_temp = AugmentedLagrangianGradient(g=g, h=h, f_grad=f_grad, jac_g=jac_g, jac_h=jac_h, lam_temp=u,
                                                 mu_temp=v, pen_temp=pen, m=m, n=n, p=p)

        new_aug_lag_grad = augLagGrad_temp(x_new)
        norm_expression = new_aug_lag_grad - previous_aug_lag_grad + scale * (
                x - x_new)
        # print(grad_norm)

        solved = (inner_tol <= tol_dual and sub_solved) and np.linalg.norm(norm_expression, np.inf) <= tol_prim
        tired = tot_it >= maxiter
        can_stop = solved or tired or broken
        """
        if np.linalg.norm(norm_expression, np.inf) < tol and tot_it > 0:
            print(f"Success! Converged in {tot_it} iterations with {tot_inner_it} inner iterations.")
            #print("Norm expression ", norm_expression)
            print(f"Norm of inequality constraints: {ineq_norm}, equality constraints: {eq_norm}.")
            return x_new
        """

        if not can_stop:
            # Penalty update
            if ineq_norm > tau * previous_ineq_norm or eq_norm > tau * previous_eq_norm:
                pen = np.minimum(gamma * pen, pen_max)
            # Decrease inner tolerance
            inner_tol = np.maximum(kappa_tol * inner_tol,tol_dual)
        #print(pen)
        # set up the constraint norms for the next loop
        previous_ineq_norm = ineq_norm
        previous_eq_norm = eq_norm
        x = x_new
        previous_aug_lag_grad = new_aug_lag_grad
    if solved:
        return OptimizationResult(x = x,
                              success = True,
                              status = 0,
                              message = f"Success! Converged in {tot_it} iterations with {tot_inner_it} inner iterations.",
                              nnz=int(np.count_nonzero(x)),
                              obj = f(x) + phi(x),
                              outer_iterations=tot_it,
                              inner_iterations=tot_inner_it,
                              ineq_violation=float(np.linalg.norm(np.minimum(g_new, 0),ord=np.inf)),
                              eq_violation=float(np.linalg.norm(h_new,ord=np.inf)),
        )
    elif tired:
        return OptimizationResult(x=x,
                              success=False,
                              status=1,
                              message=f"Failure! Maximum iterations reached.",
                              nnz=int(np.count_nonzero(x)),
                              obj=f(x) + phi(x),
                              outer_iterations=tot_it,
                              inner_iterations=tot_inner_it,
                              ineq_violation=float(np.linalg.norm(np.minimum(g_new, 0),ord=np.inf)),
                              eq_violation=float(np.linalg.norm(h_new,ord=np.inf)),
                              )
    elif broken:
        return OptimizationResult(x=x,
                              success=False,
                              status=2,
                              message=f"Failure! Stepsize gamma became to small.",
                              nnz=int(np.count_nonzero(x)),
                              obj = f(x) + phi(x),
                              outer_iterations=tot_it,
                              inner_iterations=tot_inner_it,
                              ineq_violation=float(np.linalg.norm(np.minimum(g_new, 0),ord=np.inf)),
                              eq_violation=float(np.linalg.norm(h_new,ord=np.inf)),
                              )
    else:
        return OptimizationResult(x=x,
                              success=False,
                              status=3,
                              message="Unknown",
                              nnz=int(np.count_nonzero(x)),
                              obj = f(x) + phi(x),
                              outer_iterations=tot_it,
                              inner_iterations=tot_inner_it,
                              ineq_violation=float(np.linalg.norm(np.minimum(g_new, 0),ord=np.inf)),
                              eq_violation=float(np.linalg.norm(h_new,ord=np.inf)),
                              )
