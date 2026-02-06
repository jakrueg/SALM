import numpy as np
from pylops.utils import NDArray
from pyproximal import ProxOperator
from pyproximal.ProxOperator import _check_tau


class MCP(ProxOperator):
    r"""MCP proximal operator.

        Proximal operator of the minimax concave penalty function.:

        Parameters
        ----------
        sigma : :obj:`float` or :obj:`np.ndarray` or :obj:`func`, optional
            Multiplicative coefficient of MCP. This can be a constant number, a list
            of values (for 2-dimensional inputs, acting on the second dimension) or
            a function that is called passing a counter which keeps track of how many
            times the ``prox`` method has been invoked before and returns a scalar (or a list of)
            ``sigma`` to be used.
        g : :obj:`np.ndarray`, optional
            Vector to be subtracted
        a : :obj: `float`
            Parameter of the MCP function, must be larger than 2.
    """
    def __init__(self, sigma=1., g=None, a = 3.7):
        super().__init__(None,False)
        self.sigma = sigma
        self.g = g
        self.a = a


    def __call__(self, x: NDArray) -> float:
        return float(np.sum(self.elementwise(x)))

    def elementwise(self, x: NDArray) -> NDArray:
        f = self.sigma * np.ones_like(x)
        absx = np.abs(x)
        ind = absx <= self.a
        f[ind] = self.sigma*((2 * absx[ind] * self.a - x[ind] ** 2) / (self.a ** 2))
        return f




    @_check_tau
    def prox(self, x, tau):
        gamma = tau*self.sigma
        if self.a**2 < 2*gamma:
            x1 = x.copy()
            x1[np.square(x) <= 2*gamma] = 0
            return x1
        elif self.a**2 > 2*gamma:
            thresh = (2*gamma)/self.a
            abs_x = np.abs(x)
            sign_x = np.sign(x)
            x2 = x.copy()
            ind = abs_x <= thresh
            x2[ind] = 0
            ind = np.logical_and(thresh <= abs_x, abs_x <= self.a)
            x2[ind] = (self.a/(self.a**2-2*gamma))*self.a*x[ind]-2*gamma*sign_x[ind]
            return x2
        else:
            x3 =x.copy()
            x3[np.abs(x3) <= self.a] = 0
            return x3

