# This is a sample Python script.
import SALM
import numpy as np
import portfolio as pf
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def phi(v):
        return 0.0


    def phi_prox(v, alpha):
        return v
    # print_hi('PyCharm')



    # Example test case
    def f(x):
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2  # Quadratic objective


    def f_grad(x):
        return np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])  # Gradient of f(x)


    def h(x):
        return x[0] + x[1] - 3  # Equality constraint

    def grad_h(x):
        return np.array([1,1])

    # Initial guess
    x0 = np.array([0.0, 0.0])

    # Call the solver
    #result = SALM.salm(f, phi=None, x0=x0, h=[h], f_grad=f_grad,jac_h=[grad_h],phi_prox=phi_prox, tol=1e-6, maxiter=200)
    #print("Optimized solution:", result)



# Example test case with multiple constraints
    def f(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2  # Quadratic objective

    def f_grad(x):
        return np.array([2 * (x[0] - 1), 2 * (x[1] - 2), 2 * (x[2] - 3)])  # Gradient of f(x)

    def h(x):
        return x[0] + x[1] + x[2] - 6  # Equality constraint

    def g1(x):
        return x[0] - 2  # Inequality constraint 1

    def g2(x):
        return x[1] - 1  # Inequality constraint 2

    # Jacobians for the constraints
    def jac_h(x):
        return np.array([1, 1, 1])  # Jacobian of equality constraint h(x)

    def jac_g1(x):
        return np.array([1, 0, 0])  # Jacobian of inequality constraint g1(x)

    def jac_g2(x):
        return np.array([0, 1, 0])  # Jacobian of inequality constraint g2(x)

    # Initial guess
    x0 = np.array([2.0, 1.0, 3.0])

    # Call the solver
    result = SALM.salm(f, phi=phi,phi_prox=phi_prox, x0=x0, h=[h], g=[g1, g2], f_grad=f_grad, jac_h=[jac_h], jac_g=[jac_g1, jac_g2], tol=1e-5, maxiter=200)
    print("Optimized solution:", result)


    # Rosenbrock example
    def phi(x):
        return np.abs(x[0])

    def phi_prox(x,alpha):
        return np.array([np.sign(x[0])*np.maximum(0.0,np.abs(x[0])-alpha),x[1]])
    def f(x):
        return 10*(x[1]+1-(x[0]+1)**2)**2

    def f_grad(x):
        return  np.array([-40*(x[0]+1)*(-x[0]**2-2*x[0]+x[1]),20*(-1.0*(x[0]+1)**2+x[1]+1)]) # Gradient of f(x)


    def g1(x):
        return x[0] +x[1]  # Inequality constraint 1

    def g2(x):
        return x[0] -x[1]  # Inequality constraint 2

    # Jacobians for the constraints
    def jac_g1(x):
        return np.array([1, 1])  # Jacobian of inequality constraint g1(x)

    def jac_g2(x):
        return np.array([1, -1])  # Jacobian of inequality constraint g2(x)

    # Initial guess
    x0 = np.array([1.0,1.0])

    # Call the solver
    result = SALM.salm(f, phi=phi,phi_prox=phi_prox, x0=x0, g=[g1, g2], f_grad=f_grad, jac_g=[jac_g1, jac_g2], tol=1e-5, maxiter=200)
    print("Optimized solution:", result)


