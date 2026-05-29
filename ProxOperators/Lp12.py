import numpy as np
from pylops.utils import NDArray
from pyproximal import ProxOperator
from pyproximal.ProxOperator import _check_tau


class Lp12(ProxOperator):
    r"""l^p proximal operator for p = 1/2.

        Proximal operator of the l^p quasinorm with p = 1/2

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
        return float(self.sigma * np.sum(np.sqrt(np.abs(x))))





    @_check_tau
    def prox(self, x, tau):
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

        return x1

