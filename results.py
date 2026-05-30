# results.py
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


import numpy as np
import pickle

import pandas as pd

# dataclass for unified optimization result format
@dataclass
class OptimizationResult:
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
    lam: np.ndarray
    mu: np.ndarray

# helper function to build problem path
def build_path(filename,solver):
    return Path("Data") / f"{filename}_{solver}_result.pkl"

# loads all portfolio data of one solver into a DataFrame
def load_portfolio(solver):
    path = Path("/home/krueger/MV/size200")
    problem_files=sorted(path.glob("*.mat"))
    num_probs = len(problem_files)

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

# loads portfolio result data of a single portfolio specified by solver and problem name
def load_single_portfolio(solver,prob_name):
    filepath = build_path(prob_name, solver)
    with open(filepath, "rb") as f:
        opt_res = pickle.load(f)

    return opt_res

