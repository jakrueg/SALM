# Example: using problem classification system
import pycutest
import numpy as np
import SALM
import proxGenerator
import scipy.optimize as opt


# Find unconstrained, variable-dimension problems
linear_probs = pycutest.find_problems(objective='linear', constraints='linear quadratic general')
quadratic_probs = pycutest.find_problems(objective='quadratic', constraints='linear quadratic general')
sum_of_squares_probs = pycutest.find_problems(objective='sum of squares', constraints='linear quadratic general')
other_probs = pycutest.find_problems(objective='other', constraints='linear quadratic general')

print(f"Linear {len(linear_probs)}")
print(f"Quadratic {len(quadratic_probs)}")
print(f"Sum of squares {len(sum_of_squares_probs)}")
print(f"Other {len(other_probs)}")

print(linear_probs[:10])

prob = pycutest.import_problem('A4X12')


mp = len(prob.is_eq_cons)
p = sum(prob.is_eq_cons)
m = sum(1-prob.is_eq_cons)
print(f"{mp} constraints with {m} inequality and {p} equality")
#[eq_start, eq_end, ineq_start, ineq_end] =   [0,p,p,mp] #if prob.eq_cons_first else [m,mp,0,m]


#print(prob.is_eq_cons)
#print(prob.eq_cons_first)
#print(f"Equality constraints from {eq_start} to {eq_end}")
#print(f"Inequality constraints from {ineq_start} to {ineq_end}")
#c=(prob.cons(prob.x0,index=0, gradient=True)[-1])[eq_start:eq_end]
#print(c)


def f(x):
    return prob.obj(x)

def f_grad(x):
    return prob.grad(x)


def g(x):
    return prob.cons(x)[~prob.is_eq_cons]

def h(x):
    return prob.cons(x)[prob.is_eq_cons]

def jac_g(x):
    return (prob.cons(x,gradient=True)[-1])[~prob.is_eq_cons]

def jac_h(x):
    return (prob.cons(x,gradient=True)[-1])[prob.is_eq_cons]

desc = "null"

x0 = prob.x0

constraints = [
    {
        "type": "eq",
        "fun": h,
        "jac": jac_h
    },
    {
        "type": "ineq",
        "fun": g,
        "jac": jac_g
    }
]
result2 = opt.minimize(fun=f,
                       x0=x0,
                       jac=f_grad,
                       constraints=constraints,
                       method="trust-constr",
                       options={
                           # "verbose": 3,
                           "gtol": 1e-8
                       })

print("Solution (Trust Region):\n", result2.x)
print("Objective (Trust Region):\n", prob.obj(result2.x))

x0 = result2.x
# Call the solver
result = SALM.salm(f=f, phi=proxGenerator.get_phi(desc), phi_prox=proxGenerator.get_phi_prox(desc),x0 = x0, g=g, h=h, f_grad=f_grad, jac_g=jac_g,jac_h=jac_h, tol_grad=1e-3,
               maxiter=200)
print("Optimized solution (SALM):\n", result)
print("Objective (SALM):\n", prob.obj(result))


