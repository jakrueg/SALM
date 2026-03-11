import numpy as np
from pylops.utils import NDArray
from pyproximal import L1
from pyproximal.ProxOperator import _check_tau


class L1Slack(L1):
    r"""L1 proximal operator for CUTEst test problems

        Proximal operator for the special setting of regularized CUTEst problems. It is split into three parts.
         - x-part: indicator function of a cartesian product of given closed intervals
         - a-part: indicator function of cartesian product of the positive real numbers plus L1-regularization
         - b-part: L1-regularization

        Parameters
        ----------
        n : int
            length of x-part
        m : int
            length of a-part
        p : int
            length of b-part
        bl : np.ndarray
            vector of lower bounds for x-part
        bu : np.ndarray
            vector of upper bounds for x-part
        sigma : float, optional
            regularization parameter


    """
    # initialize
    def __init__(self, n,m,p, bl,bu, sigma=1.):
        super().__init__(sigma = sigma, g = None)
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.sigma = sigma
        self.n = n
        self.m = m
        self.p = p
        self.bl = bl
        self.bu = bu

    # function call of phi
    def __call__(self, x: NDArray) -> float:
        return super().__call__(x= x[self.n:self.n + self.m+self.p])


    # prox evaluation
    @_check_tau
    def prox(self, x, tau):
        st = self.sigma * tau
        x_ret = np.copy(x)
        # projection on bounds in x-part
        x_ret[0:self.n] = np.maximum(np.minimum(x[0:self.n],self.bu),self.bl)

        # projection onto [0,\infty) and L1-Prox for a-part
        for i in range(0,self.m):
            if x[self.n + i] < st:
                x_ret[self.n + i] = x[self.n + i]**2 / (2*st)
            else:
                x_ret[self.n + i] = x[self.n + i]-st

        # soft thresholding for b-part
        x_ret[self.n+self.m:self.n + self.m + self.p] = super().prox(x[self.n+self.m:self.n + self.m+self.p],tau)
        return x_ret

