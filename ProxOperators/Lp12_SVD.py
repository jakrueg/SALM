import numpy as np
from pylops.utils import NDArray
from pyproximal import ProxOperator
from pyproximal.ProxOperator import _check_tau


class MCP_Lp12(ProxOperator):
    r"""l^(1/2) proximal operator.

        Proximal operator of the l^p proximal operator for p = 1/2 applied elementwise to a vector of singular values of a given matrix.
        Parameters
        ----------
        sigma : float, optional
            regularization parameter
        g : :obj:`np.ndarray`, optional
            Vector to be subtracted
    """
    def __init__(self, sigma=1., g=None):
        super().__init__(None,False)
        self.sigma = sigma
        self.g = g

    def __call__(self, x: NDArray) -> float:
        U,diag,Vh = np.linalg.svd(x, full_matrices=False)
        return float(self.sigma * np.sum(np.sqrt(np.abs(diag))))





    @_check_tau
    def prox(self, W, tau):
        U,x,Vh = np.linalg.svd(W, full_matrices=False)
        gamma = tau*self.sigma
        abs_x = np.abs(x)
        # return array
        x1 = np.zeros_like(x)

        # threshold for which the prox is not zero
        thresh = 1.5* gamma ** (2/3)

        mask = (abs_x > thresh) & (x > 0)

        c = (-gamma/4.0) * (3.0/abs_x[mask])**1.5

        phi = (2.0/3.0) * np.arccos(c)

        x1[mask] = (2.0/3.0)*x[mask]*(1+np.cos(phi))

        return U @ np.diag(x1) @ Vh

    def subdiff(self, W):   
            U,x,Vh = np.linalg.svd(W, full_matrices=False)
            subdiff_Lp12_vec =  np.zeros_like(x)
            mask = (np.abs(x) > 0.0)
            subdiff_Lp12_vec[mask] = np.sign(x[mask]) * self.sigma*(1/(2*np.sqrt(np.abs(x[mask]))))
    
            return U @ np.diag(subdiff_Lp12_vec) @ Vh

