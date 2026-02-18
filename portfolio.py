from pathlib import Path

import L0Box
import SALM
import numpy as np
import scipy.optimize as opt
import gurobipy as gp
from gurobi_optimods.portfolio import MeanVariancePortfolio
import pandas as pd
import zipfile
import io
import requests
import warnings

import pyproximal
import MCP

#import SALM_PANOC
#import proxGenerator
warnings.simplefilter(action="ignore", category=FutureWarning)


# Random Portfolio Generation
def generate_portfolio_random(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Sigma = A.T @ A / n + 1e-2 * np.eye(n)  # SPD covariance
    mu = rng.uniform(0.01, 0.2, size=n)  # expected returns
    return Sigma, mu


def build_path(path, filename, extension):
    return Path(path) / f"{filename}.{extension}"


def read_covariance_matrix(path, filename): # input .mat file
    filepath = build_path(path, filename, "mat")
    with open(filepath, 'r') as file:
        # Read dimension
        n_temp = int(file.readline().strip())

        # Read the remaining lines into a NumPy array
        matrix = np.loadtxt(file)

    # Safety checks
    if matrix.shape != (n_temp, n_temp):
        raise ValueError(f"Expected a {n_temp}x{n_temp} matrix, but got {matrix.shape}")

    return matrix

def read_mu(path, filename): # input .txt file
    filepath = build_path(path, filename, "txt")
    with open(filepath, 'r') as file:
        n_temp = int(file.readline())
        v1, v2 = np.loadtxt(file, unpack=True)
    return v1

def read_ub(path, filename): # input .bds file
    filepath = build_path(path, filename, "bds")
    with open(filepath, 'r') as file:
        v1, v2 = np.loadtxt(file, unpack=True)
    return v2


def read_rho(path, filename): # input .rho file
    filepath = build_path(path, filename, "rho")
    with open(filepath, 'r') as file:
        rho_temp = float(file.readline())
    return rho_temp

def read_n(path, filename):
    filepath = build_path(path, filename, "txt")
    with open(filepath, 'r') as file:
        n_temp = int(file.readline())
    return n_temp

"""
def generate_portfolio_famafrench():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_CSV.zip"

    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    df = pd.read_csv(
        z.open(z.namelist()[0]),
        skiprows=11,
        index_col=0
    )

    df = df[df.index.astype(str).str.match(r"^\d{6}$")]

    df.index = pd.to_datetime(df.index, format="%Y%m")

    industries = df.apply(pd.to_numeric) / 100.0
    returns = industries
    Sigma_ret = returns.cov().values
    mu_ret = returns.mean().values
    return Sigma_ret, mu_ret
"""

#n = 25
tol_grad = 1e-8
tol_cons =1e-6
#Sigma, mu = generate_portfolio_famafrench()
#n = len(mu)

path = "/home/jannis/Documents/MV/size200"
filename = "orl200-05-a"
Sigma = read_covariance_matrix(path, filename)
mu = read_mu(path, filename)
rho = read_rho(path, filename)
ub = read_ub(path, filename)
n = read_n(path, filename)

"""
n = 40
Sigma, mu = generate_portfolio_random(n)
#print(f"Sigma: {Sigma}")
rho = 1e-3
"""
#regu = 1/n



phi = L0Box.L0Box(sigma = 1,u = ub)
#phi = pyproximal.L0(sigma =1.0)

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

m = gp.Model()

x_gp = m.addMVar(len(mu), lb = 0, ub = 1, name = "x")

bin_mask = m.addMVar(len(mu), vtype = gp.GRB.BINARY, name = "b")
# Budget constraint: all investments sum up to 1
m.addConstr(x_gp.sum() == 1, "Budget_Constraint")

# Lower bound on expected return
m.addConstr(mu @ x_gp >= rho, "Minimal_Return")

# Force x to 0 if not traded; see formula above
m.addConstr(x_gp <= bin_mask, name="Indicator")

# Minimum position: if invested, at least min_weight
min_weight = 0.002  # e.g., at least 2% per asset
m.addConstr(x_gp >= min_weight * bin_mask, "Min_Position")


# Define objective function: Minimize risk
m.setObjective(0.5 * x_gp @ Sigma @ x_gp, gp.GRB.MINIMIZE)

#phi = pyproximal.L0(sigma =1.0)
#phi = MCP.MCP(sigma = 0.1)

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
#print(0.5* x0.T @ Sigma @ x0)
result = SALM.salm(f, phi=phi, x0=x0, h=[h], g=g, f_grad=f_grad, jac_h=[jac_h], jac_g=jac_g,
                   tol=tol_grad, maxiter=200)
result_x = result.x
print(result.message)
print("Optimized solution (SALM):\n", result_x)
print("Objective:", f(result_x))
nnz = int(np.count_nonzero((np.abs(result_x)>tol_grad)))
print(f"Non-zero (over threshold):{nnz} of {n}")
print("Return:", mu @ result_x)
print("Budget:", np.sum(result_x))

#mvp = MeanVariancePortfolio(mu,Sigma)
#pf = mvp.efficient_portfolio(rho,max_positions=nnz)
#print("Optimized solution (Gurobi):\n", pf.x)
#print("Objective:", pf.risk)
#print("Return:", pf.ret)

# Cardinality constraint: at most K positions
#cardinality_constr = m.addConstr(bin_mask.sum() <= nnz, "Cardinality")
#m.optimize()





#"""

result = SALM.salm(f, phi=phi, x0=x0, h=[h], g=g, f_grad=f_grad, jac_h=[jac_h], jac_g=jac_g,
                   tol=tol_grad, maxiter=200, subsolver='PANOC_plus')
result_x = result.x
#result = SALM_PANOC.salm(f, phi=phi,  x0=x0, h=[h], g=g, f_grad=f_grad, jac_h=[jac_h], jac_g=jac_g,
#                   tol_grad=tol_grad,tol_cons=tol_cons, maxiter=200)
print(result.message)
print("Optimized solution (SALM-PANOC+):\n", result_x)
print("Objective:", f(result_x))
nnz = int(np.count_nonzero((np.abs(result_x)>tol_grad)))
print(f"Non-zero (over threshold):{nnz} of {n}")
print("Return:", mu @ result_x)
print("Budget:", np.sum(result_x))


cardinality_constr = m.addConstr(bin_mask.sum() <= nnz, "Cardinality")

# Set the warm-start
x_gp.VarHintVal = result_x
x_gp.VarHintPri = 5
bin_mask.VarHintVal = (result_x > 1e-6).astype(float)
bin_mask.VarHintPri = 5

m.update()
m.optimize()

print("Optimized solution (Gurobi):\n", x_gp.X)
print(f"Minimum Risk:     {m.ObjVal:.6f}")
print(f"Expected return:  {mu @ x_gp.X:.6f}")
print(f"Solution time:    {m.Runtime:.2f} seconds\n")
print(f"Number of trades: {sum(bin_mask.X)}\n")

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
