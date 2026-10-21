from Algorithms import SALM
import numpy as np

import ProxOperators.ZeroPenalty
from ProxOperators import MCP_SVD
import pyproximal

def random_orthonormal(n, r, rng=None):
    rng = np.random.default_rng(rng)
    G = rng.standard_normal((n, r))      # n x r standard Gaussian
    Q, R = np.linalg.qr(G, mode="reduced")
    # Optional: fix signs so Q is Haar-distributed (uniform on the Stiefel manifold)
    Q = Q * np.sign(np.diag(R))
    return Q


delta = 0.3

def generate_problem(n=100, r = 20):
    """
    Generate a least-squares problem on the sphere.

    Minimize ||X - W||_F^2 subject to ||X||_F^2 = R^2
    where W is a given matrix and R is a given radius.
    """
    mu = 0.7
    R = 3
    lam = 2.0

    rng = np.random.default_rng(0)
    U = random_orthonormal(n, r, rng)
    V = random_orthonormal(n, r, rng)

    singular_values = np.random.uniform(low=0.5, high=2, size=r)

    correction_factor = R/np.linalg.norm(singular_values)
    singular_values *= correction_factor

    X = U @ np.diag(singular_values) @ V.T

    subdiff_MCP_vec =  np.zeros_like(singular_values)
    mask = np.logical_and(singular_values > 0.0, singular_values < delta)
    subdiff_MCP_vec[mask] = 2/delta - 2*singular_values[mask]/(delta**2)

    W = (1+mu) * X + 0.5*lam* U @ np.diag(subdiff_MCP_vec) @ V.T

    X0 = R*W/np.linalg.norm(W, 'fro')
    return W, R, X0,lam, X


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

W, R, X0, lam, X_ref = generate_problem(n, r)


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


res = SALM.salm(
    f=objective,
    phi=VectorizedMatrixProx(
        MCP_SVD.MCP_SVD(sigma=lam, delta=delta),
        W.shape,
    ),
    #phi = VectorizedMatrixProx(
    #    ProxOperators.ZeroPenalty.ZeroPenalty(),
    #    W.shape,
    #),
    x0=X0.reshape(-1),
    g=[],
    h=[eq_constraints],
    f_grad=gradient,
    jac_g=None,
    jac_h=eq_jacobian,
    tol=1e-6,
    maxiter=1000,
)

print("Status:", res.status)
print("Message:", res.message)
print("Objective value:", objective(res.x))
print("Equality constraint violation:", eq_constraints(res.x))
print("Iterations:", res.outer_iterations)
print("rank of solution:", np.linalg.matrix_rank(as_matrix(res.x)))
print("Distance to reference solution:", np.linalg.norm(as_matrix(res.x) - X_ref, 'fro'))