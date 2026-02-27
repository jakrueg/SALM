

class Jacobian:
    def __init__(self,jac,jac_prod):
        self.jac = jac
        self.jac_prod = jac_prod

    def __call__(self, x):
        return self.jac(x)

    def jacobian_vector_product(self,x,vec):
        return self.jac_prod(x,vec)