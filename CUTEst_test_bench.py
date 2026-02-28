import pickle
import resource

import cyipopt

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
#import scipy.optimize as opt


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

problematic = []

def run_problem_set(pset):
    #pset_return = []
    for problem_name in pset:
        if problem_name not in problematic:

            problem = pycutest.import_problem(problem_name)
            num_cons = np.maximum(sum(problem.is_eq_cons),sum(1-problem.is_eq_cons))
            if num_cons <=1000 and 1000 >= problem.n > num_cons:
                run_cutest_problem(problem)
            pycutest.clear_cache(problem_name)
    #return pset_return

#linear_probs_filtered = filter_problem_set(linear_probs)
#quadratic_probs_filtered = filter_problem_set(quadratic_probs)
#sum_of_squares_probs_filtered = filter_problem_set(sum_of_squares_probs)
#other_probs_filtered = filter_problem_set(other_probs)

#print(len(linear_probs_filtered))
#print(len(sum_of_squares_probs_filtered))

print(linear_probs[:20])
#print(unconstrained_probs[:20])

class IPOPTAdapter:
    def __init__(self, prob):
        self.prob = prob

    # Objective function
    def objective(self, x):
        return self.prob.obj(x)

    # Gradient of objective
    def gradient(self, x):
        return self.prob.obj(x, gradient=True)[1]

    # Constraint vector
    def constraints(self, x):
        return self.prob.cons(x)

    # Jacobian of constraints (flattened row-major)
    def jacobian(self, x):
        return self.prob.cons(x, gradient=True)[1].flatten()

    # (Optional but recommended) Jacobian structure
    def jacobianstructure(self):
        m = self.prob.m
        n = self.prob.n
        rows, cols = np.nonzero(np.ones((m, n)))
        return rows, cols

def run_cutest_problem(prob):

    n = prob.n
    m_plus_p = len(prob.is_eq_cons)
    p = sum(prob.is_eq_cons)
    m = sum(1-prob.is_eq_cons)
    print(prob.name)
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
            return -prob.cons(x_temp)[~prob.is_eq_cons]+a_temp


        def jac_g_matrix(x_slack):
            x_temp = x_slack[0:n]
            #a_temp = x_slack[n:n + m]

            x_part = -(prob.cons(x_temp, gradient=True)[-1])[~prob.is_eq_cons]
            a_part = np.eye(m)
            if p>0:
                b_part = np.zeros((m,p))
                return np.hstack((x_part,a_part,b_part))
            return np.hstack((x_part,a_part))

        def jac_g_product(x_slack,vec):
            x_temp = x_slack[0:n]
            vec_full = np.zeros(m+p)
            vec_full[~prob.is_eq_cons] = vec

            x_part = -prob.jprod(x=x_temp, transpose = True, p=vec_full)
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
    adapter = IPOPTAdapter(prob)
    problem_ipopt = cyipopt.Problem(
        n=n,
        m=prob.m,
        problem_obj=adapter,
        lb = prob.bl,
        ub = prob.bu,
        cl = prob.cl,
        cu = prob.cu
    )
    #desc = "null"
    #phi = pyproximal.L1(sigma=1.0)


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
    result2 = minimize_ipopt(fun=prob.obj,
                           x0=prob.x0,
                           jac=prob.grad,
                           constraints=constraints,
                           #method="trust-constr",
                           )
    """
    ipopt_x , info = problem_ipopt.solve(prob.x0)
    #print(result2.__dict__)
    #print(info)
    #print("Multipliers:\n",result2.v)
    print("Solution (IPOPT):\n", ipopt_x)
    print("Objective (IPOPT):\n", prob.obj(ipopt_x))
    print("Constraints:\n",prob.cons(ipopt_x))

    with open(f"CUTEst/linear/{prob.name}_IPOPT_result.pkl", "wb") as write_file:
        pickle.dump(info, write_file)

    #all_multipliers = np.hstack([arr.ravel() for arr in result2.v])
    all_multipliers = info['mult_g']
    max_abs_multiplier = np.max(np.abs(all_multipliers))
    #print(max_abs_multiplier)
    regularization = 10.0+max_abs_multiplier
    print(f"Lower bound: {prob.bl}, upper bound: {prob.bu}")
    phi = L1Slack.L1Slack(n= n, m=m,p=p,sigma = regularization,bl=prob.bl,bu=prob.bu)

    # a0 = np.ones(m)/m
    # a0 = np.zeros(m)
    # b0 = np.ones(p)/p
    a0 = prob.cons(prob.x0)[~prob.is_eq_cons]
    # print(a0)
    b0 = -prob.cons(prob.x0)[prob.is_eq_cons]
    # print(g(np.concatenate((prob.x0,a0,b0))))
    x0 = np.concatenate((prob.x0, a0, b0), axis=0)
    #x0 = np.concatenate((ipopt_x, a0, b0), axis=0)
    # print(len(x0))


    # Call the solver
    result = SALM.salm(f=f, phi=phi, x0 = x0, g=g, h=h, f_grad=f_grad, jac_g=jac_g, jac_h=jac_h, tol=1e-6,
                       maxiter=50,subsolver="PANOC_plus")
    result_x = result.x[0:n]
    print(result.message)
    print("Optimized solution (SALM):\n", result_x)
    print("Objective (SALM):\n", prob.obj(result_x))
    print("Regularization:\n",phi(result_x))
    print("Constraints:\n",-prob.cons(result_x))


    with open(f"CUTEst/linear/{prob.name}_SALM_result.pkl", "wb") as write_file:
        pickle.dump(result, write_file)
    return result

prob_test = pycutest.import_problem('KIWCRESC')
print(prob_test.cons(prob_test.x0))
ret_result = run_cutest_problem(prob_test)
#run_problem_set(linear_probs)