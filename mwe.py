import numpy as np

from panoc_plus import panoc_plus
import pyproximal
from sklearn.linear_model import Lasso



# Problem dimensions
np.random.seed(0)
m, n = 200, 100

A = np.random.randn(m, n) / np.sqrt(m)
x_true = np.zeros(n)
x_true[:10] = np.random.randn(10)   # sparse signal
b = A @ x_true + 0.01 * np.random.randn(m)

print(x_true)

lam = 0.1

# Smooth part
def f(Ax):
    return 0.5 * np.dot(Ax - b, Ax - b)

def grad_f(Ax):
    return Ax - b

# ℓ1 prox
def prox_l1(y, gamma):
    z = np.sign(y) * np.maximum(np.abs(y) - gamma * lam, 0.0)
    return z, lam * np.linalg.norm(z, 1)

# Lipschitz constant
L = np.linalg.norm(A, 2) ** 2
gamma = 0.95 / L

phi = pyproximal.L1(sigma=lam)

# Run PANOC+
res = panoc_plus(
    fun=f,
    grad=grad_f,
    phi = phi,
    x0=np.zeros(n),
    A=A,
    #gamma=gamma,
    verbose=True,
)

print("Recovered support:", np.count_nonzero(res.x))
print("Error ||x-x_true||:", np.linalg.norm(res.x - x_true))

clf = Lasso(alpha=lam, fit_intercept=False, max_iter=10000)
clf.fit(A, b)

print("Sklearn error:", np.linalg.norm(clf.coef_ - x_true))
