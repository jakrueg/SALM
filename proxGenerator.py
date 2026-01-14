import numpy as np

regu = 1/30
tol = 1e-6

def get_phi(desc):
    match desc:
        case "l1":
            return lambda v: np.linalg.norm(v,ord=1)
        case "l0":
            return lambda v: np.count_nonzero(v)
        case "l0_scaled":
            return lambda v: np.count_nonzero((v>tol))*regu
        case "l0_tol":
            return lambda v: np.count_nonzero((v>tol))
        case _:
            return lambda v: 0.0


def get_phi_prox(desc):
    match desc:
        case "l1":
            return lambda v, alpha: np.sign(v) * np.maximum(np.abs(v) - alpha, 0)
        case "l0":
            return lambda v, alpha: v * (v ** 2 > 2 * alpha)
        case "l0_tol":
            return lambda v, alpha: v * (v ** 2 > 2 * alpha)
        case "l0_scaled":
            return lambda v, alpha: v * (v**2 > 2 * alpha*regu)
        case _:
            return lambda v,alpha: v

def make_panoc_prox(desc):
    """
    Returns a function prox(y, gamma) -> (z, g(z))
    compatible with PANOC+.
    """
    phi = get_phi(desc)
    phi_prox = get_phi_prox(desc)

    def prox(y, gamma):
        z = phi_prox(y, gamma)
        g_z = phi(z)
        return z, g_z

    return prox