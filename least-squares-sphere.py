from Algorithms import SALM
import csv
import json
import numpy as np
from pathlib import Path
from time import perf_counter

import ProxOperators.ZeroPenalty
from ProxOperators import MCP_SVD, Lp12_SVD
import pyproximal

def random_orthonormal(n, r, rng=None):
    rng = np.random.default_rng(rng)
    G = rng.standard_normal((n, r))      # n x r standard Gaussian
    Q, R = np.linalg.qr(G, mode="reduced")
    # Optional: fix signs so Q is Haar-distributed (uniform on the Stiefel manifold)
    Q = Q * np.sign(np.diag(R))
    return Q

delta = 1.5
lam = 0.1
regularizer = MCP_SVD.MCP_SVD(sigma=lam, delta=delta)
#regularizer = Lp12_SVD.MCP_Lp12(sigma=3.3*lam)

def generate_random_problem(n=100, r=20, seed=0):
    """
    Generate a least-squares problem on the sphere.

    Minimize ||X - W||_F^2 subject to ||X||_F^2 = R^2
    where W is a given matrix and R is a given radius.
    """
    R = 3
  

    rng = np.random.default_rng(seed)
    U = random_orthonormal(n, r, rng)
    V = random_orthonormal(n, r, rng)

    singular_values = rng.uniform(low=0.5, high=2, size=r)

    correction_factor = R/np.linalg.norm(singular_values)
    singular_values *= correction_factor

    W = U @ np.diag(singular_values) @ V.T


    X0 = R*W/np.linalg.norm(W, 'fro')
    return W, R, X0

def generate_problem(n=100, r=20, seed=0):
    """
    Generate a least-squares problem on the sphere.

    Minimize ||X - W||_F^2 subject to ||X||_F^2 = R^2
    where W is a given matrix and R is a given radius.
    """
    mu = 0.7
    R = 3
  

    rng = np.random.default_rng(seed)
    U = random_orthonormal(n, r, rng)
    V = random_orthonormal(n, r, rng)

    singular_values = rng.uniform(low=0.5, high=2, size=r)

    correction_factor = R/np.linalg.norm(singular_values)
    singular_values *= correction_factor

    X = U @ np.diag(singular_values) @ V.T


    W = (1+2*mu) * X + regularizer.subdiff(X)

    X0 = R*W/np.linalg.norm(W, 'fro')
    return W, R, X0, X


class VectorizedMatrixProx:
    """Adapt a matrix proximal operator to SALM's vector decision variables."""

    def __init__(self, prox_operator, matrix_shape):
        self.prox_operator = prox_operator
        self.matrix_shape = matrix_shape
        self.prox = self._prox

    def __call__(self, x):
        return self.prox_operator(np.asarray(x).reshape(self.matrix_shape))

    def _prox(self, x, tau):
        matrix = np.asarray(x).reshape(self.matrix_shape)
        return self.prox_operator.prox(matrix, tau).reshape(-1)


n= 100
r = 20

output_dir = Path(__file__).resolve().parent / "csv" / "ls-sphere"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "results.csv"
csv_fields = [
    "seed",
    "status",
    "message",
    "objective_value",
    "equality_constraint_violation",
    "outer_iterations",
    "inner_iterations",
    "wall_clock_seconds",
    "rank_of_solution",
    "final_multipliers",
    "rank_tolerance",
    "count_above_tolerance",
]

with output_path.open("w", newline="", encoding="utf-8") as csv_file:
    csv.DictWriter(csv_file, fieldnames=csv_fields).writeheader()

for seed in range(10):
    #W, R, X0, X_ref = generate_problem(n, r,seed=seed)
    W,R, X0 = generate_random_problem(n, r, seed=seed)

    phi = VectorizedMatrixProx(regularizer, W.shape)
    
    def as_matrix(x):
        return np.asarray(x).reshape(W.shape)


    def objective(x):
        X = as_matrix(x)
        return 0.5 * np.linalg.norm(X - W, 'fro')**2


    def gradient(x):
        return (as_matrix(x) - W).reshape(-1)


    def eq_constraints(x):
        X = as_matrix(x)
        return np.linalg.norm(X, 'fro')**2 - R**2


    def eq_jacobian(x):
        return (2 * as_matrix(x).reshape(-1))[None, :]

    tol = 1e-6
    start_time = perf_counter()
    res = SALM.salm(
        f=objective,
        phi=phi,
        x0=X0.reshape(-1),
        g=[],
        h=[eq_constraints],
        f_grad=gradient,
        jac_g=None,
        jac_h=eq_jacobian,
        tol=tol,
        maxiter=500,
        #subsolver="PANOC_plus"
    )
    wall_clock_seconds = perf_counter() - start_time

    mu_final = res.mu
    #kkt_residual = 

    composite_func = lambda x: objective(x) + phi(as_matrix(x))
    
    print(f"---------------Seed {seed}:-----------------")
    print("Status:", res.status)
    print("Message:", res.message)
    print("Objective value:", objective(res.x))
    #print("Reference objective value:", composite_func(X_ref))
    print("Equality constraint violation:", eq_constraints(res.x))
    print("Outer iterations:", res.outer_iterations)
    print("Inner iterations:", res.inner_iterations)
    print("Wall-clock time (seconds):", wall_clock_seconds)
    print("rank of solution:", np.linalg.matrix_rank(as_matrix(res.x)))
    print("final multipliers:", mu_final)
    #print("Distance to reference solution:", np.linalg.norm(as_matrix(res.x) - X_ref, 'fro'))
    print("-----")
    Y = as_matrix(res.x)
    s = np.linalg.svd(Y, compute_uv=False)
    #tol = s[0] * max(Y.shape) * np.finfo(s.dtype).eps

    print("rank tolerance:", tol)
    #print("singular values:", s)
    print("count above tolerance:", np.count_nonzero(s > tol))

    row = {
        "seed": seed,
        "status": res.status,
        "message": res.message,
        "objective_value": objective(res.x),
        "equality_constraint_violation": eq_constraints(res.x),
        "outer_iterations": res.outer_iterations,
        "inner_iterations": res.inner_iterations,
        "wall_clock_seconds": wall_clock_seconds,
        "rank_of_solution": np.linalg.matrix_rank(Y),
        "final_multipliers": json.dumps(np.asarray(mu_final).tolist()),
        "rank_tolerance": tol,
        "count_above_tolerance": np.count_nonzero(s > tol),
    }
    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        csv.DictWriter(csv_file, fieldnames=csv_fields).writerow(row)
        csv_file.flush()