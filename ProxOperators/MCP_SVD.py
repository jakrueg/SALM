import numpy as np
from pylops.utils import NDArray
from pyproximal import ProxOperator
from pyproximal.ProxOperator import _check_tau


class MCP_SVD(ProxOperator):
    r"""MCP proximal operator.

        Proximal operator of the minimax concave penalty function applied elementwise to a vector of singular values of a given matrix.
        Parameters
        ----------
        sigma : float, optional
            regularization parameter
        g : :obj:`np.ndarray`, optional
            Vector to be subtracted
        delta : float
            Parameter of the MCP function, must be larger than zero.
    """
    def __init__(self, sigma=1., g=None, delta = 2.0):
        super().__init__(None,False)
        self.sigma = sigma
        self.g = g
        self.delta = delta


    def __call__(self, x: NDArray) -> float:
        U,diag,Vh = np.linalg.svd(x, full_matrices=False)
        return float(np.sum(self.elementwise(diag)))    

    def elementwise(self, x: NDArray) -> NDArray:
        f = self.sigma * np.ones_like(x)
        absx = np.abs(x)
        ind = absx <= self.delta
        f[ind] = self.sigma*((2 * absx[ind] * self.delta - x[ind] ** 2) / (self.delta ** 2))
        return f




    @_check_tau
    def prox(self, W, tau):
        U,x,Vh = np.linalg.svd(W, full_matrices=False)
        gamma = tau*self.sigma
        if self.delta**2 < 2*gamma:
            x1 = x.copy()
            x1[np.square(x) <= 2*gamma] = 0
            return U @ np.diag(x1) @ Vh
        elif self.delta**2 > 2*gamma:
            thresh = (2*gamma)/self.delta
            abs_x = np.abs(x)
            sign_x = np.sign(x)
            x2 = x.copy()
            ind = abs_x <= thresh
            x2[ind] = 0
            ind = np.logical_and(thresh <= abs_x, abs_x <= self.delta)
            x2[ind] = (self.delta / (self.delta ** 2 - 2 * gamma)) * (self.delta * x[ind] - 2 * gamma * sign_x[ind])
            return U @ np.diag(x2) @ Vh
        else:
            x3 =x.copy()
            x3[np.abs(x3) <= self.delta] = 0
            return U @ np.diag(x3) @ Vh

