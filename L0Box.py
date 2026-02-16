import numpy as np
from pylops.utils import NDArray
from pyproximal import ProxOperator
from pyproximal.ProxOperator import _check_tau


class L0Box(ProxOperator):
    r"""L0Box proximal operator.

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
    def __init__(self, sigma=1., g=None, u = None):
        super().__init__(None,False)
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        if np.any(u < 0):
            raise ValueError("u must have non-negative entries")
        self.sigma = sigma
        self.g = g
        self.u = u
        self.n = len(u)

    def __call__(self, x: NDArray) -> float:
        return self.sigma * np.count_nonzero(x)


    @_check_tau
    def prox(self, x, tau):
        gl2 = tau*self.sigma
        y = np.zeros_like(x)
        sqrt_gl2 = np.sqrt(gl2)

        mask1 = self.u != 0
        mask2 = x > sqrt_gl2
        mask3 = x <= self.u
        mask4 = 2 * self.u * x > self.u ** 2 + gl2

        final_mask = mask1 & mask2 & (mask3 | mask4)

        y[final_mask] = x[final_mask]
        countnzy = np.sum(final_mask)

        return self.sigma*countnzy


