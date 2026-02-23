# results.py
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


import numpy as np
import pickle

import pandas as pd


@dataclass
class OptimizationResult:
    #problem_name: str
    success: bool
    status: int
    message: str
    x: np.ndarray
    nnz: int
    outer_iterations: int
    inner_iterations: int
    ineq_violation: float
    eq_violation: float
    obj: float
    #residual_norm: float

def build_path(filename,solver):
    return Path("Data") / f"{filename}_{solver}_result.pkl"

"""
with open("Data/orl200-005-a_Gurobi_result.pkl", "rb") as f:
    result = pickle.load(f)

result_x = result.x

#result = SALM_PANOC.salm(f, phi=phi,  x0=x0, h=[h], g=g, f_grad=f_grad, jac_h=[jac_h], jac_g=jac_g,
#                   tol_grad=tol_grad,tol_cons=tol_cons, maxiter=200)

print(result.message)
print("Optimized solution (SALM-PANOC+):\n", result_x)
print("Objective:", result.obj)
print("Inner iterations:", result.inner_iterations)
#nnz = int(np.count_nonzero((np.abs(result_x) > tol)))
print(f"Non-zero (over threshold):{result.nnz} of {len(result_x)}")
#print("Return:", mu @ result_x)
print("Budget:", np.sum(result_x))
"""
def load_portfolio(solver):
    path = Path("/home/jannis/Documents/MV/size200")
    problem_files=sorted(path.glob("*.mat"))
    num_probs = len(problem_files)

    #solver = "proxGrad"
    #solver = "PANOC"
    #solver = "Gurobi"
    results = []
    for problem in problem_files:
        problem_name = problem.stem
        filepath = build_path(problem_name,solver)
        with open(filepath, "rb") as f:
            opt_res = pickle.load(f)
            results.append(
                {
                    "problem_name": problem_name,
                    "success": opt_res.success,
                    "status": opt_res.status,
                    "nnz": opt_res.nnz,
                    "outer_iterations": opt_res.outer_iterations,
                    "inner_iterations": opt_res.inner_iterations,
                    "ineq_violation": opt_res.ineq_violation,
                    "eq_violation": opt_res.eq_violation,
                    "obj": opt_res.obj
                }
            )

    df_results = pd.DataFrame(results)
    return df_results

def plot_results(df_results,metric):
    num_probs = len(df_results)

    #df_success = df_results[df_results["success"]==True]
    df_success = df_results
    #print(df)

    values = df_success[metric].values
    values_sorted = np.sort(values)
    rho = np.arange(1, num_probs+1)
    x = np.arange(1, np.max(values_sorted)+100)

    y = np.copy(x)
    for i in range(len(y)):
        y[i] = np.sum(values_sorted<=x[i])

    plt.step(x, y, where="post")
    plt.xlabel(metric)
    plt.ylabel("Number of problems solved")
    plt.ylim(ymin=0,ymax=num_probs)
    plt.title("Empirical CDF")
    plt.grid(True)
    plt.show()
