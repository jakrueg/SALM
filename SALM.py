import numpy as np
import copt
from scipy.optimize import minimize

def salm(f,
    phi,
    x0,
    g = None,
    h = None,
    f_grad=None,
    phi_prox=None,
    jac_g=None,
    jac_h=None,
    pen0 = 1,
    pen_max = 1e8,
    gamma = 2,
    lam0 = None,
    mu0=None,
    lam_max=1e8,
    mu_max=1e8,
    tol=1e-6,
    maxiter=200,
    tau = 0.5
):
    # Initialization
    x = np.asarray(x0,dtype=float)
    g = g or []
    h = h or []
    n = x.size
    m = len(g)
    p = len(h)
    # initialize multipliers
    if lam0 is None:
        lam = np.zeros(m)
    else:
        lam = np.asarray(lam0,dtype=float)
    if mu0 is None:
        mu = np.zeros(p)
    else:
        mu = np.asarray(mu0,dtype=float)

    pen = float(pen0)

    # returns evaluation vector of the equality constraints
    def eval_h(x_value):
        if p<1:
            return []
        l = [h_i(x_value) for h_i in h]
        return np.asarray(l)

    # returns evaluation vector of the inequality constraints
    def eval_g(x_value):
        if m<1:
            return []
        return np.asarray([g_i(x_value) for g_i in g])

    # returns evaluation matrix for the jacobian of h
    def eval_jac_h(x_value):
        if not jac_h:
            return None
        if callable(jac_h):
            return np.atleast_1d(jac_h(x_value))
        return np.vstack([np.atleast_1d(jac_h_i(x_value)) for jac_h_i in jac_h])

    # returns evaluation matrix for the jacobian of g
    def eval_jac_g(x_value):
        if not jac_g:
            return None
        if callable(jac_g):
            return np.atleast_1d(jac_g(x_value))
        return np.vstack([np.atleast_1d(jac_g_i(x_value)) for jac_g_i in jac_g])


    # evaluates the augmented lagrangian function
    def augmented_lagrangian(x_value,lam_temp,mu_temp,pen_temp):
        f_val = float(f(x_value))+float(phi(x_value))
        #f_val = float(f(x_value))
        g_temp = eval_g(x_value)
        h_temp = eval_h(x_value)
        eq_term = np.dot(mu_temp,h_temp)+0.5*pen_temp*np.dot(h_temp,h_temp) if len(h_temp) > 0 else 0.0
        g_pos = np.maximum(0.0,g_temp)
        ineq_term = np.dot(lam_temp,g_pos)+0.5*pen_temp*np.dot(g_pos,g_pos) if len(g_temp) > 0 else 0.0
        return f_val + ineq_term + eq_term

    # computes the gradient of the differentiable part of the Augmented Lagrangian
    def augmented_lagrangian_grad(x_value,lam_temp,mu_temp,pen_temp):
        if f_grad is None:
            return None
        f_grad_val = np.asarray(f_grad(x_value),dtype=float)
        grad_sum = f_grad_val.copy()
        h_temp = eval_h(x_value)
        #print(h_temp)
        h_Jac_temp = eval_jac_h(x_value)
        if  len(h_temp) > 0:
            grad_sum += np.transpose(np.matmul(np.transpose(h_Jac_temp),mu_temp+pen_temp*h_temp))
        g_temp = eval_g(x_value)
        g_Jac_temp = eval_jac_g(x_value)
        #print(g_Jac_temp)
        if len(g_temp) > 0:
            #grad_sum += g_Jac_temp.T.dot(lam_temp+pen_temp*np.maximum(0.0,g_temp))
            grad_sum += np.transpose(np.matmul(np.transpose(g_Jac_temp),lam_temp+pen_temp*np.maximum(0.0,g_temp)))
        #print(grad_sum)
        return grad_sum


    # SALM Loop

    previous_ineq_norm = np.inf
    previous_eq_norm = np.inf
    u = np.clip(lam,0.0,lam_max)
    v = np.clip(mu,-mu_max,mu_max)

    for k in range(maxiter):
        # define inner function for solution
        def inner_fun(x_value):
            return augmented_lagrangian(x_value,u,v,pen)
        def inner_jac(x_value):
            return augmented_lagrangian_grad(x_value, u, v, pen)
        if phi_prox is not None:
            res = copt.minimize_proximal_gradient(fun=inner_fun,jac=inner_jac,x0=x,verbose=2,tol=1e-6)
            #print(res.x)
            #print(res.nit)
            #print(res.success)
        else:
            res  = minimize(inner_fun, x, jac=inner_jac, method='L-BFGS-B')
            print("No suitable prox-operator!")
            break
        x_new = res.x
        if m>0:
            g_new = eval_g(x_new)
            lam_new = np.maximum(0.0, u + pen * g_new)
            ineq_norm = np.linalg.norm(np.minimum(-g_new, (1 / pen) * u))
        else:
            ineq_norm = 0.0

        if p>0:
            h_new = eval_h(x_new)
            mu_new = np.maximum(0.0,v+pen*h_new)
            eq_norm = np.linalg.norm(h_new)
        else:
            eq_norm = 0.0

        # Termination criteria
        grad_norm = np.linalg.norm(augmented_lagrangian_grad(x_new, u, v, pen))
        #print(grad_norm)
        if grad_norm < tol and eq_norm < tol and ineq_norm < tol:
            print(f"Converged in {k} iterations.")
            return x_new
        # Penalty update
        if ineq_norm > tau*previous_ineq_norm or eq_norm > tau*previous_eq_norm:
            pen = np.minimum(gamma * pen,pen_max)

        # set up the constraint norms for the next loop
        previous_ineq_norm = ineq_norm
        previous_eq_norm = eq_norm
        x=x_new

    return 0