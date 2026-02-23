from scipy.sparse import lil_matrix
import numpy as np

class AugmentedLagrangianGradient:
    def __init__(self,g,h,f_grad, jac_g,jac_h, lam_temp, mu_temp, pen_temp, m, n,p):
        self.g = g
        self.h = h
        self.f_grad = f_grad
        self.jac_g = jac_g
        self.jac_h = jac_h
        self.lam_temp = lam_temp
        self.mu_temp = mu_temp
        self.pen_temp = pen_temp
        self.m = m
        self.n = n
        self.p = p
        self.J_g = lil_matrix((m, n))  # allocated once
        self.J_h = lil_matrix((p, n))

    # returns evaluation vector of the equality constraints
    def eval_h(self, x_value):
        if callable(self.h):
            return np.atleast_1d(self.h(x_value))
        if self.p < 1:
            return []
        l = [h_i(x_value) for h_i in self.h]
        return np.asarray(l)

    # returns evaluation vector of the inequality constraints
    def eval_g(self, x_value):
        if callable(self.g):
            return np.atleast_1d(self.g(x_value))
        if self.m < 1:
            return []
        return np.asarray([g_i(x_value) for g_i in self.g])

    def update_jacobian(self, J, jac, x):
        for i in range(J.shape[0]):
            J.rows[i].clear()
            J.data[i].clear()

        for i, jac_i in enumerate(jac):
            row = jac_i(x)
            if hasattr(row, "rows"):  # sparse lil_matrix
                J.rows[i].extend(row.rows[0])
                J.data[i].extend(row.data[0])
            else:  # dense numpy array
                row = np.atleast_1d(row)
                nz = np.nonzero(row)[0]
                J.rows[i].extend(nz.tolist())
                J.data[i].extend(row[nz].tolist())



    def __call__(self, x_value):
        self.update_jacobian(J=self.J_g, jac=self.jac_g, x=x_value)
        self.update_jacobian(J=self.J_h, jac=self.jac_h, x=x_value)
        if self.f_grad is None:
            return None
        f_grad_val = np.asarray(self.f_grad(x_value), dtype=float)
        grad_sum = f_grad_val.copy()
        h_temp = self.eval_h(x_value)
        #print(h_temp)
        h_Jac_temp = self.J_h.tocsr()
        hT = h_Jac_temp.transpose()
        if len(h_temp) > 0:
            c = hT @ h_temp
            grad_sum += np.transpose(hT @ np.atleast_1d(self.mu_temp + self.pen_temp * h_temp))
        g_temp = self.eval_g(x_value)
        g_Jac_temp = self.J_g.tocsr()
        #print(g_Jac_temp)
        if len(g_temp) > 0:
            #grad_sum += g_Jac_temp.T.dot(lam_temp+pen_temp*np.maximum(0.0,g_temp))
            grad_sum += np.transpose(g_Jac_temp.transpose() @( self.lam_temp + self.pen_temp * np.maximum(0.0, g_temp)))
        #print(grad_sum)
        return grad_sum