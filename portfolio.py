import SALM
import numpy as np
import scipy.optimize as opt
import pandas_datareader.data as web
import warnings

import SALM_PANOC
import proxGenerator
warnings.simplefilter(action="ignore", category=FutureWarning)


# Random Portfolio Generation
def generate_portfolio_random(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Sigma = A.T @ A / n + 1e-2 * np.eye(n)  # SPD covariance
    mu = rng.uniform(0.01, 0.2, size=n)  # expected returns
    return Sigma, mu


def generate_portfolio_famafrench():
    industries = web.DataReader(
        "30_Industry_Portfolios",
        "famafrench"
    )[0] / 100.0
    returns = industries
    Sigma_ret = returns.cov().values
    mu_ret = returns.mean().values
    return Sigma_ret, mu_ret

#phi_desc = "l0"
phi_desc = "l0_tol"
#phi_desc = "l0_scaled"
#phi_desc = "l1"
#n = 25
tol_grad = 1e-4
tol_cons =1e-6
Sigma, mu = generate_portfolio_famafrench()
n = len(mu)
#Sigma, mu = generate_portfolio_random(n)
rho = 1e-3

regu = 1/n

def f(x):
    return 0.5 * x.T @ Sigma @ x  # Quadratic objective


def f_grad(x):
    return Sigma @ x  # Gradient of f(x)


def h(x):
    return np.sum(x) - 1.0  # Equality constraint


def g0(x):
    return rho - np.dot(mu, x)  # minimum expected return


# Jacobians for the constraints
def jac_h(x):
    return np.ones(n)  # Jacobian of equality constraint h(x)


def jac_g0(x):
    return -1 * mu  # Jacobian of inequality constraint g0(x)

# l0 norm


g = [g0]
jac_g = [jac_g0]

for i in range(n):
    def g_temp(x, i=i):
        return -x[i]


    def jac_g_temp(x, i=i):
        erg = np.concatenate(-np.eye(1, n, k=i))
        #print(i)
        #print(erg)
        return erg


    g.append(g_temp)
    jac_g.append(jac_g_temp)
"""
for i in range(n):
    def g_temp(x):
        return x[i]-u
    def jac_g_temp(x):
        return np.eye(1,n,k=i)
    g.append(g_temp)
    jac_g.append(jac_g_temp)
"""
#print(jac_g[1](x_test))
#print(np.vstack([np.atleast_1d(jac_g_i(x_test)) for jac_g_i in jac_g]))

x0 = np.ones(n) / n

result = SALM.salm(f, phi=proxGenerator.get_phi(phi_desc), phi_prox=proxGenerator.get_phi_prox(phi_desc), x0=x0, h=[h], g=g, f_grad=f_grad, jac_h=[jac_h], jac_g=jac_g,
                   tol_grad=tol_grad,tol_cons=tol_cons, maxiter=200)
print("Optimized solution (SALM):\n", result)
print("Objective:", f(result))
print(f"Non-zero (over threshold):{np.count_nonzero((np.abs(result)>tol_grad))} of {n}")
print("Return:", mu @ result)
print("Budget:", np.sum(result))

#"""
result = SALM_PANOC.salm(f, phi=proxGenerator.get_phi(phi_desc), phi_prox=proxGenerator.make_panoc_prox(phi_desc), x0=x0, h=[h], g=g, f_grad=f_grad, jac_h=[jac_h], jac_g=jac_g,
                   tol_grad=tol_grad,tol_cons=tol_cons, maxiter=200)
print("Optimized solution (SALM-PANOC+):\n", result)
print("Objective:", f(result))
print(f"Non-zero (over threshold):{np.count_nonzero((np.abs(result)>tol_grad))} of {n}")
print("Return:", mu @ result)
print("Budget:", np.sum(result))
#"""
def g_return(x):
    return np.dot(mu, x) - rho


def jac_g_return(x):
    return mu


bounds = [(0.0, None)] * n
constraints = [
    {
        "type": "eq",
        "fun": h,
        "jac": jac_h
    },
    {
        "type": "ineq",
        "fun": g_return,
        "jac": jac_g_return
    }
]
result2 = opt.minimize(fun=f,
                       x0=x0,
                       jac=f_grad,
                       constraints=constraints,
                       bounds=bounds,
                       method="trust-constr",
                       options={
                           # "verbose": 3,
                           "gtol": 1e-8
                       })

print("Solution (Trust Region):\n", result2.x)
print("Objective:", result2.fun)
print("Return:", mu @ result2.x)
print("Budget:", np.sum(result2.x))
