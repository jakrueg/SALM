import numpy as np

class LBFGSOperator:
    def __init__(self, M, x):
        """
        Class for implementing a L-BFGS algorithm

        Parameters
        ----------
        M : int
            memory size
        x : np.ndarray
            reference vector
        """
        self.M =M
        self.currmem  = 0
        self.curridx = 0

        self.s = np.zeros_like(x)
        self.y = np.zeros_like(x)

        self.s_M = [np.zeros_like(x) for _ in range(M)]
        self.y_M = [np.zeros_like(x) for _ in range(M)]

        self.ys_M = np.zeros(M, dtype=x.dtype)
        self.alphas = np.zeros(M, dtype=x.dtype)

        self.H = 1.0

    def update(self, s, y):
        self.s[:] = s
        self.y[:] = y

        ys = float(np.dot(self.s, self.y))

        if ys > 0.0:
            self.curridx += 1
            if self.curridx >= self.M:
                self.curridx = 0

            self.currmem = min(self.currmem + 1, self.M)

            self.ys_M[self.curridx] = ys
            self.s_M[self.curridx][:] = self.s
            self.y_M[self.curridx][:] = self.y

            yty = float(np.dot(self.y, self.y))
            self.H = ys / yty

        return self

    def reset(self):
        self.currmem = 0
        self.curridx = 0
        self.H = 1.0

    def __matmul__(self, v):
        d = np.empty_like(v)
        self.mul(d, v)
        return d

    def __call__(self, v):
        return self @ v

    def mul(self, d, v):
        d[:] = v
        idx = self._loop1(d)
        d *= self.H
        self._loop2(d, idx)
        return d

    def _loop1(self, d):
        idx = self.curridx
        for _ in range(self.currmem):
            self.alphas[idx] = np.dot(self.s_M[idx], d) / self.ys_M[idx]
            d -= self.alphas[idx] * self.y_M[idx]

            idx -= 1
            if idx < 0:
                idx = self.M - 1
        return idx

    def _loop2(self, d, idx):
        for _ in range(self.currmem):
            idx += 1
            if idx >= self.M:
                idx = 0

            beta = np.dot(self.y_M[idx], d) / self.ys_M[idx]
            d += (self.alphas[idx] - beta) * self.s_M[idx]

        return d
