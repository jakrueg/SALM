

class Jacobian:
    r"""Auxiliary class for Jacobian for memory efficiency.

            Parameters
            ----------
            jac : callable
                Jacobian function
            jac_prod : callable
                implementation of matrix - vector product with transposed Jacobian
        """
    def __init__(self,jac,jac_prod):
        self.jac = jac
        self.jac_prod = jac_prod

    def __call__(self, x):
        return self.jac(x)

    def jacobian_vector_product(self,x,vec):
        return self.jac_prod(x,vec)