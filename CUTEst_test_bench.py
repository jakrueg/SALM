import resource

from cyipopt.scipy_interface import _minimize_ipopt_iv

# Limit to 24 GB
memory_limit = 24 * 1024 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))


# Example: using problem classification system
import numpy as np
import pycutest
#import pyproximal

from ProxOperators import L1Slack
from Algorithms import SALM
from Utilities import jacobian
import scipy.optimize as opt
from cyipopt import minimize_ipopt

unconstrained_probs = pycutest.find_problems(objective="quadratic", constraints="unconstrained")


linear_probs = pycutest.find_problems(objective='linear', constraints='linear quadratic general',m=[0,1000])
quadratic_probs = pycutest.find_problems(objective='quadratic', constraints='linear quadratic general',m=[0,1000])
sum_of_squares_probs = pycutest.find_problems(objective='sum of squares', constraints='linear quadratic general',m=[0,1000])
other_probs = pycutest.find_problems(objective='other', constraints='linear quadratic general',m=[0,1000])

print(f"Linear {len(linear_probs)}")
print(f"Quadratic {len(quadratic_probs)}")
print(f"Sum of squares {len(sum_of_squares_probs)}")
print(f"Other {len(other_probs)}")
print(f"Unconstrained {len(unconstrained_probs)}")

def filter_problem_set(pset):
    pset_return = []
    for problem_name in pset:
        problem = pycutest.import_problem(problem_name)
        num_cons = np.maximum(sum(problem.is_eq_cons),sum(1-problem.is_eq_cons))
        if num_cons <=1000 and num_cons < problem.n:
            pset_return.append(problem_name)
    return pset_return

linear_probs_filtered = filter_problem_set(linear_probs)
#quadratic_probs_filtered = filter_problem_set(quadratic_probs)
sum_of_squares_probs_filtered = filter_problem_set(sum_of_squares_probs)
#other_probs_filtered = filter_problem_set(other_probs)

print(len(linear_probs_filtered))
print(len(sum_of_squares_probs_filtered))

print(other_probs[:20])
#print(unconstrained_probs[:20])

prob = pycutest.import_problem('STANCMIN')

n = prob.n
m_plus_p = len(prob.is_eq_cons)
p = sum(prob.is_eq_cons)
m = sum(1-prob.is_eq_cons)
print(f"Dimension is {n}, {m_plus_p} (calculated) {prob.m} (reported) constraints with {m} inequality and {p} equality")
#[eq_start, eq_end, ineq_start, ineq_end] =   [0,p,p,mp] #if prob.eq_cons_first else [m,mp,0,m]


#print(prob.is_eq_cons)
#print(prob.eq_cons_first)
#print(f"Equality constraints from {eq_start} to {eq_end}")
#print(f"Inequality constraints from {ineq_start} to {ineq_end}")
#c=(prob.cons(prob.x0,index=0, gradient=True)[-1])[eq_start:eq_end]
#print(c)

#------------------------------------------------
# Convention: x_slack = [x,a,b] with len(x)=n, len(a)=m, len(b)=p
def f(x_slack):
    x_temp = x_slack[0:n]
    return prob.obj(x_temp)

def f_grad(x_slack):
    x_temp = x_slack[0:n]
    return np.concatenate((prob.grad(x_temp),np.zeros(m+p)))
"""
result1 = panoc_plus.panoc_plus(f,f_grad,proxGenerator.make_panoc_prox(""),prob.x0)

print(result1.x)
print(prob.obj(result1.x))
print(np.linalg.norm(prob.grad(result1.x)))


result2 = opt.minimize(fun=f,
                       x0=prob.x0,
                       jac=f_grad,
                       options={
                           # "verbose": 3,
                           "gtol": 1e-6
                       })

print(result2.x)
print(prob.obj(result2.x))
print(np.linalg.norm(prob.grad(result2.x)))

"""
constraints = []

if m < 1:
    # only equality constraints
    g = []
    jac_g = []
else:
    def g(x_slack):
        x_temp = x_slack[0:n]
        a_temp = x_slack[n:n+m]
        return prob.cons(x_temp)[~prob.is_eq_cons]+a_temp


    def jac_g_matrix(x_slack):
        x_temp = x_slack[0:n]
        #a_temp = x_slack[n:n + m]

        x_part = (prob.cons(x_temp, gradient=True)[-1])[~prob.is_eq_cons]
        a_part = np.eye(m)
        if p>0:
            b_part = np.zeros((m,p))
            return np.hstack((x_part,a_part,b_part))
        return np.hstack((x_part,a_part))

    def jac_g_product(x_slack,vec):
        x_temp = x_slack[0:n]
        vec_full = np.zeros(m+p)
        vec_full[~prob.is_eq_cons] = vec

        x_part = prob.jprod(x=x_temp, transpose = True, p=vec_full)
        b_part = np.zeros(p)

        return np.concatenate((x_part, vec,b_part))

    jac_g = jacobian.Jacobian(jac = jac_g_matrix,jac_prod=jac_g_product)


    constraints.append({
        "type": "ineq",
        "fun": lambda x : -prob.cons(x)[~prob.is_eq_cons],
        "jac": lambda x : -(prob.cons(x, gradient=True)[-1])[~prob.is_eq_cons]
    })




if p < 1:
    # only inequality constraints
    h = []
    jac_h = []
else:
    def h(x_slack):
        x_temp = x_slack[0:n]
        b_temp = x_slack[n+m:n+m+p]
        return prob.cons(x_temp)[prob.is_eq_cons] + b_temp


    def jac_h_matrix(x_slack):
        x_temp = x_slack[0:n]

        x_part = (prob.cons(x_temp, gradient=True)[-1])[prob.is_eq_cons]
        b_part = np.eye(p)
        if m>0:
            a_part = np.zeros((p,m))
            return np.hstack((x_part,a_part,b_part))
        return np.hstack((x_part,b_part))

    def jac_h_product(x_slack,vec):
        x_temp = x_slack[0:n]
        vec_full = np.zeros(m+p)
        vec_full[prob.is_eq_cons] = vec
        x_part = prob.jprod(x=x_temp,transpose=True,p = vec_full)

        a_part = np.zeros(m)
        return np.concatenate((x_part, a_part,vec))


    jac_h = jacobian.Jacobian(jac=jac_h_matrix,jac_prod=jac_h_product)

    constraints.append({
        "type": "eq",
        "fun": lambda x: -prob.cons(x)[prob.is_eq_cons],
        "jac": lambda x: -(prob.cons(x, gradient=True)[-1])[prob.is_eq_cons]
    })


#desc = "null"
#phi = pyproximal.L1(sigma=1.0)

a0 = np.ones(m)/m
b0 = np.ones(p)/p
#a0 = prob.cons(prob.x0)[~prob.is_eq_cons]
#b0 = prob.cons(prob.x0)[prob.is_eq_cons]
x0 = np.concatenate((prob.x0,a0,b0),axis=0)
#print(len(x0))
"""
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
"""
result2 = minimize_ipopt(fun=prob.obj,
                       x0=prob.x0,
                       jac=prob.grad,
                       constraints=constraints,
                       method="trust-constr",
                       options={
                           # "verbose": 3,
                           "gtol": 1e-8
                       })
#print(dir(result2))
#print("Multipliers:\n",result2.v)
print("Solution (Trust Region):\n", result2.x)
print("Objective (Trust Region):\n", prob.obj(result2.x))
print("Constraints:\n",prob.cons(result2.x))


all_multipliers = np.hstack([arr.ravel() for arr in result2.v])
max_abs_multiplier = np.max(np.abs(all_multipliers))
#print(max_abs_multiplier)
regularization = 10.0+max_abs_multiplier
phi = L1Slack.L1Slack(n= n, m_plus_p=m+p,sigma = regularization)

#x0 = result2.x
# Call the solver
result = SALM.salm(f=f, phi=phi, x0 = x0, g=g, h=h, f_grad=f_grad, jac_g=jac_g, jac_h=jac_h, tol=1e-6,
                   maxiter=200,subsolver="STANCMIN")
result_x = result.x[0:n]
print(result)
print("Optimized solution (SALM):\n", result_x)
print("Objective (SALM):\n", prob.obj(result_x))
print("Constraints:\n",prob.cons(result_x))


