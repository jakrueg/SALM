#import resource
# Limit memory to 24 GB to avoid memory overflow
#memory_limit = 24 * 1024 * 1024 * 1024
#resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))


from multiprocessing import Process
import pickle
from pathlib import Path

from ProxOperators import MCP
from Algorithms import SALM
import numpy as np
import gurobipy as gp
import warnings

from ProxOperators.Lp12 import Lp12
from Utilities import gurobi_adapter

warnings.simplefilter(action="ignore", category=FutureWarning)


# random portfolio generation
def generate_portfolio_random(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Sigma = A.T @ A / n + 1e-2 * np.eye(n)  # SPD covariance
    mu = rng.uniform(0.01, 0.2, size=n)  # expected returns
    return Sigma, mu

# helper function for building path to problem files
def build_path(path, filename, extension):
    return Path(path) / f"{filename}.{extension}"

# reads covariance matrix data from one file
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

# reads mean value data from one file
def read_mu(path, filename): # input .txt file
    filepath = build_path(path, filename, "txt")
    with open(filepath, 'r') as file:
        n_temp = int(file.readline())
        v1, v2 = np.loadtxt(file, unpack=True)
    return v1

# reads upper bounds from one file
def read_ub(path, filename): # input .bds file
    filepath = build_path(path, filename, "bds")
    with open(filepath, 'r') as file:
        v1, v2 = np.loadtxt(file, unpack=True)
    return v2

# reads minimum return threshold from one file
def read_rho(path, filename): # input .rho file
    filepath = build_path(path, filename, "rho")
    with open(filepath, 'r') as file:
        rho_temp = float(file.readline())
    return rho_temp

# reads problem dimension from one file
def read_n(path, filename):
    filepath = build_path(path, filename, "txt")
    with open(filepath, 'r') as file:
        n_temp = int(file.readline())
    return n_temp

tol = 1e-6
tol_cons =1e-6


# Path to Portfolio Data
path = Path("/home/krueger/MV/size200")


def run_problem(problem_name,solvers,write_files = False):
    """
    Reads result data from a problem class and returns its performance statistics

    Parameters
    ----------
    problem_name : str
        string of portfolio problem name
    solvers : list of strings
        contains the names of all solvers that should run the problem
    write_files : bool
        flag that indicates if problem data should be written in .pkl files for later use

    Returns
    -------
    ret : list
        list of optimization results of the specified algorithms
    """
    algoGurobi = False
    algoProxGrad = False
    algoPANOC = False


    if "Gurobi" in solvers:
        algoGurobi = True
    if "proxGrad" in solvers:
        algoProxGrad = True
    if "PANOC" in solvers:
        algoPANOC = True


    # Read Problem Data
    print(problem_name)
    Sigma = read_covariance_matrix(path, problem_name)
    mu = read_mu(path, problem_name)
    rho = read_rho(path, problem_name)
    n = read_n(path, problem_name)
    x0 = np.ones(n) / n
    if "pard200" in problem_name:
        regularization_factor = 50
    elif "orl200-05" in problem_name:
        regularization_factor = 90
    else:
        regularization_factor = 25

    # Set up optimization problem
    phi = MCP.MCP(sigma = regularization_factor,delta = 0.1)
    #phi = Lp12(sigma = regularization_factor)

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


    g = [g0]
    jac_g = [jac_g0]

    ret = {}

    if algoProxGrad:
        # SALM with proximal gradient as subproblem solver
        result = SALM.salm(f, phi=phi, x0=x0, h=[h], g=g, f_grad=f_grad, jac_h=[jac_h], jac_g=jac_g, tol=tol, maxiter=200)
        if write_files:
            with open(f"Data/{problem_name}_proxGrad_result.pkl", "wb") as write_file:
                pickle.dump(result, write_file)
        ret["proxGrad"] = result
    if algoGurobi:
        # Gurobi Reference solver
        m = gp.Model()

        x_gp = m.addMVar(len(mu), lb=0, name="x")
        # additional variable in MIQP formulation
        xi = m.addMVar(len(mu), lb=0, ub=1, name="xi")
        # Budget constraint: all investments sum up to 1
        m.addConstr(x_gp.sum() == 1, "Budget_Constraint")

        # Lower bound on expected return
        m.addConstr(mu @ x_gp >= rho, "Minimal_Return")

        # Mixed Integer reformulation complementarity
        m.addConstr(xi * x_gp == 0, "Complementarity_Constraint")

        gurobi_scale = regularization_factor
        # Define objective function: Minimize risk
        m.setObjective(0.5 * x_gp @ Sigma @ x_gp + gurobi_scale * (len(mu) - xi.sum()), gp.GRB.MINIMIZE)
        m.setParam('TimeLimit', 60*10)
        m.optimize()
        result_gurobi = gurobi_adapter.build_result(m,x_gp.X, Sigma, regularization_factor)
        if write_files:
            with open(f"Data/{problem_name}_Gurobi_result.pkl", "wb") as write_file3:
                pickle.dump(result_gurobi, write_file3)
        ret["gurobi"] = result_gurobi
    if algoPANOC:
        # SALM with PANOC+ as subproblem solver
        result2 = SALM.salm(f, phi=phi, x0=x0, h=[h], g=g, f_grad=f_grad, jac_h=[jac_h], jac_g=jac_g,
                           tol=tol, maxiter=200, subsolver='PANOC_plus')
        if write_files:
            with open(f"Data/{problem_name}_PANOC_result.pkl", "wb") as write_file2:
                pickle.dump(result2, write_file2)
        ret["PANOC"] = result2
    return ret


def eval_all_probs(solvers,write_files):
    """
    Evaluates all portfolio problems in separate processes

    Parameters
    ----------
    solvers : list of strings
        contains the names of all solvers that should run the problem
    write_files : bool
        flag that indicates if problem data should be written in .pkl files for later use

    Returns
    -------
    ret : list
        list of optimization results of the specified algorithms
    """
    problem_files = sorted(path.glob("*.mat"))
    for problem in problem_files:
        p = Process(target=run_problem, args=(problem.stem,solvers,write_files))
        p.start()
        p.join()
