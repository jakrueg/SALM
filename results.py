# results.py
from dataclasses import dataclass
import numpy as np
import pickle

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


with open("Data/orl200-05-a_PANOC_result.pkl", "rb") as f:
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

