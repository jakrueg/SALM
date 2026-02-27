import numpy as np
from pylops.utils import NDArray
from pyproximal import L1
from pyproximal.ProxOperator import _check_tau


class L1Slack(L1):
    r"""L1 proximal operator for the slack variables

        Proximal operator of the L0-Pseudonorm with an added indicator function of the interval [0,u]:

        Parameters
        ----------
        sigma : :obj:`float` or :obj:`np.ndarray` or :obj:`func`, optional
            Multiplicative coefficient of the L0-Pseudonorm. This can be a constant number, a list
            of values (for 2-dimensional inputs, acting on the second dimension) or
            a function that is called passing a counter which keeps track of how many
            times the ``prox`` method has been invoked before and returns a scalar (or a list of)
            ``sigma`` to be used.
        g : :obj:`np.ndarray`, optional
            Vector to be subtracted
    """
    def __init__(self, n,m_plus_p, sigma=1.):
        super().__init__(sigma = sigma, g = None)
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.sigma = sigma
        self.n = n
        self.m_plus_p = m_plus_p

    def __call__(self, x: NDArray) -> float:
        return super().__call__(x= x[self.n:self.n + self.m_plus_p])


    @_check_tau
    def prox(self, x, tau):
        x_ret = np.copy(x)
        x_ret[self.n:self.n + self.m_plus_p] = super().prox(x[self.n:self.n + self.m_plus_p],tau)
        return x_ret

