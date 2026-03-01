import pickle
import resource

import cyipopt
import pandas as pd
from scipy._lib.cobyqa import problem

# Limit to 24 GB
memory_limit = 24 * 1024 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))


# Example: using problem classification system
import numpy as np
import pycutest
#import pyproximal

from pathlib import Path
from ProxOperators import L1Slack
from Algorithms import SALM
from Utilities import jacobian
from dataclasses import dataclass
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

#problematic = ["LOTSCHD","BT8","DUAL3","MWRIGHT","AVION2","HIMMELBI","QC","QCNEW","EQC","HS59","HS9","PORTFL3","LSQFIT"]
problematic = ["HS268","BT8","HIMMELBI","HS59","HS9"]


import numpy as np


class CUTEstALWrapper:
    """
    Wraps a PyCUTEst problem into:
        min f(x)
        s.t. g(x) <= 0
             h(x) = 0

    Provides:
        - f(x), grad f(x)
        - g(x), h(x)
        - Jg(x)^T v
        - Jh(x)^T v
    """

    def __init__(self, problem, inf=1e20):
        self.problem = problem
        self.n = problem.n
        self.m = problem.m

        cl = problem.cl
        cu = problem.cu

        # Storage
        eq_idx = []
        eq_rhs = []

        ineq_idx = []
        ineq_rhs = []
        ineq_sign = []

        # Classify constraints
        for i in range(self.m):
            li = cl[i]
            ui = cu[i]

            lower_finite = li > -inf
            upper_finite = ui <  inf

            # Equality
            if lower_finite and upper_finite and np.isclose(li, ui):
                eq_idx.append(i)
                eq_rhs.append(li)

            else:
                # Upper bound: c_i(x) <= ui
                if upper_finite:
                    ineq_idx.append(i)
                    ineq_rhs.append(ui)
                    ineq_sign.append(+1.0)

                # Lower bound: li <= c_i(x)
                if lower_finite:
                    ineq_idx.append(i)
                    ineq_rhs.append(li)
                    ineq_sign.append(-1.0)

        # Convert to arrays
        self.eq_idx = np.array(eq_idx, dtype=int)
        self.eq_rhs = np.array(eq_rhs)

        self.ineq_idx = np.array(ineq_idx, dtype=int)
        self.ineq_rhs = np.array(ineq_rhs)
        self.ineq_sign = np.array(ineq_sign)

        self.n_eq = len(self.eq_idx)
        self.n_ineq = len(self.ineq_idx)

    # ---------------------------------------------------
    # Objective
    # ---------------------------------------------------

    def f_grad(self, x):
        """
        Returns objective and gradient.
        Uses joint evaluation for stability.
        """
        f, g = self.problem.obj(x, gradient=True)
        return f, g

    # ---------------------------------------------------
    # Constraint evaluation
    # ---------------------------------------------------

    def h(self, x):
        """
        Equality constraints h(x) = 0
        """
        if self.n_eq == 0:
            return np.zeros(0)

        c = self.problem.cons(x)
        return c[self.eq_idx] - self.eq_rhs

    def g(self, x):
        """
        Inequality constraints g(x) <= 0
        """
        if self.n_ineq == 0:
            return np.zeros(0)

        c = self.problem.cons(x)
        values = c[self.ineq_idx]

        # sign * (c_i(x) - bound)
        return self.ineq_sign * (values - self.ineq_rhs)

    def Jg(self, x):
        """
        Returns dense Jacobian of inequality constraints:
            Jg(x) in R^{n_ineq x n}
        """
        if self.n_ineq == 0:
            return np.zeros((0, self.n))

        Jc = self.problem.jac(x)  # shape (m, n)

        rows = Jc[self.ineq_idx, :]

        # Apply sign flips for lower bounds
        return self.ineq_sign[:, None] * rows

    # ---------------------------------------------------
    # Explicit Jacobian construction
    # ---------------------------------------------------

    def Jh(self, x):
        """
        Returns dense Jacobian of equality constraints:
            Jh(x) in R^{n_eq x n}
        """
        if self.n_eq == 0:
            return np.zeros((0, self.n))

        Jc = self.problem.jac(x)  # shape (m, n)

        return Jc[self.eq_idx, :]

    # ---------------------------------------------------
    # Transposed Jacobian-vector products
    # ---------------------------------------------------

    def JhT_v(self, x, v):
        """
        Computes J_h(x)^T v

        v must have length n_eq
        Returns vector in R^n
        """
        if self.n_eq == 0:
            return np.zeros(self.n)

        if len(v) != self.n_eq:
            raise ValueError("Dimension mismatch in JhT_v")

        full = np.zeros(self.m)
        full[self.eq_idx] = v

        return self.problem.jprod(x=x, p=full, transpose=True)

    def JgT_v(self, x, v):
        """
        Computes J_g(x)^T v

        v must have length n_ineq
        Returns vector in R^n
        """
        if self.n_ineq == 0:
            return np.zeros(self.n)

        if len(v) != self.n_ineq:
            raise ValueError("Dimension mismatch in JgT_v")

        full = np.zeros(self.m)
        full[self.ineq_idx] = self.ineq_sign * v

        return self.problem.jprod(x=x, p=full, transpose=True)



def run_problem_set(pset):
    #pset_return = []
    for problem_name in pset:
        print(problem_name)
        if problem_name not in problematic:

            problem = pycutest.import_problem(problem_name)
            num_cons = np.maximum(sum(problem.is_eq_cons),sum(1-problem.is_eq_cons))
            if num_cons <=100 and 100 >= problem.n > num_cons:
                problem_data = build_cutest_problem(problem)
                run_cutest_problem(problem_data)
            pycutest.clear_cache(problem_name)
    #return pset_return

#linear_probs_filtered = filter_problem_set(linear_probs)
#quadratic_probs_filtered = filter_problem_set(quadratic_probs)
#sum_of_squares_probs_filtered = filter_problem_set(sum_of_squares_probs)
#other_probs_filtered = filter_problem_set(other_probs)

#print(len(linear_probs_filtered))
#print(len(sum_of_squares_probs_filtered))

#print(linear_probs[:20])
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

def build_cutest_problem(prob : pycutest.CUTEstProblem):

    n = prob.n
    m_plus_p = len(prob.is_eq_cons)
    p = sum(prob.is_eq_cons)
    #m = sum(1-prob.is_eq_cons)
    print(prob.name)
    #[eq_start, eq_end, ineq_start, ineq_end] =   [0,p,p,mp] #if prob.eq_cons_first else [m,mp,0,m]

    constraint_wrapper = CUTEstALWrapper(prob)
    m = constraint_wrapper.n_ineq
    print(f"Dimension is {n}, {m_plus_p} (calculated) {prob.m} (reported) constraints with {m} inequality and {p} equality")
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
        x_temp = np.array(x_slack[0:n], dtype = np.float64, order = 'C', copy = True)
        """
        print(prob.n)
        print(x_temp.shape)
        print(x_temp.dtype)
        print(x_temp.flags)
        x0 = np.array(prob.x0, dtype=np.float64, order="C", copy=True)
        print(f"Objective: {prob.obj(x0,gradient=True)}")
        print("Success!!")
        """
        return np.concatenate((prob.obj(x_temp,gradient=True)[1],np.zeros(m+p)))
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
            return constraint_wrapper.g(x_temp) + a_temp
            #return -prob.cons(x_temp)[~prob.is_eq_cons]+a_temp


        def jac_g_matrix(x_slack):
            x_temp = x_slack[0:n]
            #a_temp = x_slack[n:n + m]

            #x_part = -(prob.cons(x_temp, gradient=True)[-1])[~prob.is_eq_cons]
            x_part = constraint_wrapper.Jg(x_temp)
            a_part = np.eye(m)
            if p>0:
                b_part = np.zeros((m,p))
                return np.hstack((x_part,a_part,b_part))
            return np.hstack((x_part,a_part))

        def jac_g_product(x_slack,vec):
            x_temp = x_slack[0:n]
            #vec_full = np.zeros(m+p)
            #vec_full[~prob.is_eq_cons] = vec

            #x_part = -prob.jprod(x=x_temp, transpose = True, p=vec_full)
            x_part = constraint_wrapper.JgT_v(x_temp,vec)
            b_part = np.zeros(p)

            return np.concatenate((x_part, vec,b_part))

        jac_g = jacobian.Jacobian(jac = jac_g_matrix,jac_prod=jac_g_product)





    if p < 1:
        # only inequality constraints
        h = []
        jac_h = []
    else:
        def h(x_slack):
            x_temp = x_slack[0:n]
            b_temp = x_slack[n+m:n+m+p]
            return constraint_wrapper.h(x_temp) + b_temp
            #return prob.cons(x_temp)[prob.is_eq_cons] + b_temp


        def jac_h_matrix(x_slack):
            x_temp = x_slack[0:n]

            #x_part = (prob.cons(x_temp, gradient=True)[-1])[prob.is_eq_cons]
            x_part = constraint_wrapper.Jh(x_temp)
            b_part = np.eye(p)
            if m>0:
                a_part = np.zeros((p,m))
                return np.hstack((x_part,a_part,b_part))
            return np.hstack((x_part,b_part))

        def jac_h_product(x_slack,vec):
            x_temp = x_slack[0:n]
            #vec_full = np.zeros(m+p)
            #vec_full[prob.is_eq_cons] = vec
            #x_part = prob.jprod(x=x_temp,transpose=True,p = vec_full)

            x_part = constraint_wrapper.JhT_v(x_temp,vec)
            a_part = np.zeros(m)
            return np.concatenate((x_part, a_part,vec))


        jac_h = jacobian.Jacobian(jac=jac_h_matrix,jac_prod=jac_h_product)

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
    return {
        "problem_ipopt": problem_ipopt,
        "prob" : prob,
        "n" : n,
        "m" : m,
        "p" : p,
        "f" : f,
        "f_grad" : f_grad,
        "g" : g,
        "jac_g" : jac_g,
        "h" : h,
        "jac_h" : jac_h,
        "constraint_wrapper" : constraint_wrapper
    }
    #desc = "null"
    #phi = pyproximal.L1(sigma=1.0)
def calculate_multiplier(info):
    all_multipliers = info['mult_g']
    max_abs_multiplier = np.max(np.abs(all_multipliers))
    # print(max_abs_multiplier)
    regularization = 10.0 + max_abs_multiplier
    return regularization


def run_cutest_problem(problem_data
):

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
    problem_ipopt = problem_data['problem_ipopt']
    prob = problem_data['prob']
    n = problem_data['n']
    m = problem_data['m']
    p = problem_data['p']
    f = problem_data['f']
    f_grad = problem_data['f_grad']
    g = problem_data['g']
    jac_g = problem_data['jac_g']
    h = problem_data['h']
    jac_h = problem_data['jac_h']
    constraint_wrapper = problem_data['constraint_wrapper']

    ipopt_x , info = problem_ipopt.solve(prob.x0)
    #print(result2.__dict__)
    #print(info)
    #print("Multipliers:\n",result2.v)
    print("Solution (IPOPT):\n", ipopt_x)
    print("Objective (IPOPT):\n", prob.obj(ipopt_x))
    #print("Constraints:\n",prob.cons(ipopt_x))

    with open(f"CUTEst/quadratic/{prob.name}_IPOPT_result.pkl", "wb") as write_file:
        pickle.dump(info, write_file)
    #pycutest.clear_cache(prob.name)
    #all_multipliers = np.hstack([arr.ravel() for arr in result2.v])
    regularization = calculate_multiplier(info)
    print(f"Lower bound: {prob.bl}, upper bound: {prob.bu}")
    phi = L1Slack.L1Slack(n= n, m=m,p=p,sigma = regularization,bl=prob.bl,bu=prob.bu)

    # a0 = np.ones(m)/m
    # a0 = np.zeros(m)
    # b0 = np.ones(p)/p
    #a0 = prob.cons(prob.x0)[~prob.is_eq_cons]
    a0 = -constraint_wrapper.g(prob.x0)
    # print(a0)
    #b0 = -prob.cons(prob.x0)[prob.is_eq_cons]
    b0 = -constraint_wrapper.h(prob.x0)
    # print(g(np.concatenate((prob.x0,a0,b0))))
    x0 = np.concatenate((prob.x0, a0, b0), axis=0)
    #print(f"Phi: {phi(x0)}")
    #x0 = np.concatenate((ipopt_x, a0, b0), axis=0)
    # print(len(x0))


    # Call the solver
    result = SALM.salm(f=f, phi=phi, x0 = x0, g=g, h=h, f_grad=f_grad, jac_g=jac_g, jac_h=jac_h, tol=1e-6,
                       maxiter=100,subsolver="PANOC_plus")
    result_x = result.x[0:n]
    print(result.message)
    print("Optimized solution (SALM):\n", result_x)
    print("Objective (SALM):\n", prob.obj(result_x))
    print("Regularization:\n",phi(result.x))
    #print("Constraints:\n",-prob.cons(result_x))


    with open(f"CUTEst/quadratic/{prob.name}_SALM_result.pkl", "wb") as write_file:
        pickle.dump(result, write_file)
    return result

def max_constraint_violation(problem,x):
    """
    Computes maximal constraint violation directly from CUTEst bounds.

    Returns:
        scalar infinity-norm violation
    """
    if problem.m == 0:
        return 0.0

    c = problem.cons(x)
    cl = problem.cl
    cu = problem.cu

    # Lower bound violations: cl - c(x)
    lower_violation = np.maximum(cl - c, 0.0)

    # Upper bound violations: c(x) - cu
    upper_violation = np.maximum(c - cu, 0.0)

    return max(np.max(lower_violation), np.max(upper_violation))

def read_data(subfolder):
    folder = Path("CUTEst") / subfolder

    unique_prefixes = {
        file.stem.split("_", 1)[0]
        for file in folder.glob("*.pkl")
    }

    unique_prefixes = sorted(unique_prefixes)
    #print(len(unique_prefixes))
    #print(unique_prefixes)
    results = []
    for problem_name in unique_prefixes:
        cutest_prob = pycutest.import_problem(problem_name)

        problem_path = folder / f"{problem_name}_IPOPT_result.pkl"
        with open(problem_path, "rb") as f_ipopt:
            info = pickle.load(f_ipopt)
            x_opt = info["x"]
            #constr_violation = np.maximum(np.linalg.norm(np.minimum(cutest_prob.cons(x_opt)[~cutest_prob.is_eq_cons],0.0),ord=np.inf),np.linalg.norm(cutest_prob.cons(x_opt)[cutest_prob.is_eq_cons],ord=np.inf))
            constr_violation = max_constraint_violation(cutest_prob,x_opt)
            results.append(
                {
                    "problem_name": problem_name,
                    "method_name": "IPOPT",
                    "obj": info["obj_val"],
                    "constr_violation": constr_violation,
                    "norm_a_plus_b" : 0.0,
                    "status" : info["status"],
                    "status_msg": info["status_msg"],
                    "outer_iterations" : 0,
                    "inner_iterations" : 0
                }
            )
        problem_path = folder / f"{problem_name}_SALM_result.pkl"
        with open(problem_path, "rb") as f_salm:
            res_salm = pickle.load(f_salm)
            n = cutest_prob.n
            m = cutest_prob.m
            x_opt = res_salm.x[0:n]
            a_and_b = res_salm.x[n:n+m]
            norm_a_plus_b = np.linalg.norm(a_and_b,ord=np.inf)
            #constr_violation = np.maximum(np.linalg.norm(np.minimum(cutest_prob.cons(x_opt)[~cutest_prob.is_eq_cons],0.0),ord=np.inf),np.linalg.norm(cutest_prob.cons(x_opt)[cutest_prob.is_eq_cons],ord=np.inf))
            constr_violation = max_constraint_violation(cutest_prob,x_opt)
            results.append(
                {
                    "problem_name": problem_name,
                    "method_name": "SALM",
                    "obj": cutest_prob.obj(x_opt),
                    "constr_violation": constr_violation,
                    "norm_a_plus_b": norm_a_plus_b,
                    "status": res_salm.status,
                    "status_msg": res_salm.message,
                    "outer_iterations": res_salm.outer_iterations,
                    "inner_iterations": res_salm.inner_iterations
                }
            )

    df_results = pd.DataFrame(results)
    return df_results

#def read_problem_file(problem_name, solver):

#def performance_statistics():

@dataclass
class DataLine:
    problem_name: str
    method_name: str
    objective_value: float
    constraint_violation: float
    norm_a_plus_b: float
    status: int
    outer_iterations: int
    inner_iterations: int

#test_prob = "HS76"
#pycutest.clear_cache(test_prob)
#prob_test = pycutest.import_problem(test_prob)
#test_build = build_cutest_problem(prob_test)
#print(prob_test.cons(prob_test.x0))
#ret_result = run_cutest_problem(test_build)

run_problem_set(quadratic_probs)


problem_class = "quadratic"
df = read_data(problem_class)
df.to_csv(f"csv/{problem_class}.csv", index=False)
#print(df[["problem_name", "obj","constr_violation", "status","norm_a_plus_b"]])