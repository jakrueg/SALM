import pickle
import resource

import cyipopt
import pandas as pd

# Limit to 24 GB memory
memory_limit = 24 * 1024 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

import numpy as np
import pycutest

from pathlib import Path
from ProxOperators import L1Slack
from Algorithms import SALM
from Utilities import jacobian
from dataclasses import dataclass

linear_probs = pycutest.find_problems(objective='linear', constraints='linear quadratic general', m=[0, 1000])
quadratic_probs = pycutest.find_problems(objective='quadratic', constraints='linear quadratic general', m=[0, 1000])
sum_of_squares_probs = pycutest.find_problems(objective='sum of squares', constraints='linear quadratic general',
                                              m=[0, 1000])
other_probs = pycutest.find_problems(objective='other', constraints='linear quadratic general', m=[0, 1000])

print(f"Linear {len(linear_probs)}")
print(f"Quadratic {len(quadratic_probs)}")
print(f"Sum of squares {len(sum_of_squares_probs)}")
print(f"Other {len(other_probs)}")

# Problems that are filtered out due to having a faulty PyCUTEst implementation
problematic = ["HS268", "BT8", "HIMMELBI", "HS59", "HS9"]


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

        # storage
        eq_idx = []
        eq_rhs = []

        ineq_idx = []
        ineq_rhs = []
        ineq_sign = []

        # classify constraints
        for i in range(self.m):
            li = cl[i]
            ui = cu[i]

            lower_finite = li > -inf
            upper_finite = ui < inf

            # equality
            if lower_finite and upper_finite and np.isclose(li, ui):
                eq_idx.append(i)
                eq_rhs.append(li)

            else:
                # upper bound: c_i(x) <= ui
                if upper_finite:
                    ineq_idx.append(i)
                    ineq_rhs.append(ui)
                    ineq_sign.append(+1.0)

                # lower bound: li <= c_i(x)
                if lower_finite:
                    ineq_idx.append(i)
                    ineq_rhs.append(li)
                    ineq_sign.append(-1.0)

        # convert to arrays
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
    """
    Method for running a whole set of CUTEst problems identified by their problem name strings. It iterates through the
    problem names and runs on each the methods to
        - load the problem through PyCUTEst
        - build the problem for input in SALM
        - solve the problem

    Parameters
    ----------
    pset : array of strings
        array containing the problem names

    """
    for problem_name in pset:
        print(problem_name)
        if problem_name not in problematic:

            problem = pycutest.import_problem(problem_name)
            num_cons = np.maximum(sum(problem.is_eq_cons), sum(1 - problem.is_eq_cons))
            if num_cons <= 100 and 100 >= problem.n > num_cons:
                problem_data = build_cutest_problem(problem)
                run_cutest_problem(problem_data)
            pycutest.clear_cache(problem_name)


class IPOPTAdapter:
    """
        Wraps a PyCUTEst for input into IPOPT
        """
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


def build_cutest_problem(prob: pycutest.CUTEstProblem):
    """
        Method for building all necessary inputs for the SALM algorithm from a PyCUTEst problem.

        Parameters
        ----------
        prob : pycutest.CUTEstProblem
            PyCUTEst problem object

        Returns
        -------
        dict
            Dictionary containing all necessary inputs for the SALM algorithm

    """
    n = prob.n
    m_plus_p = len(prob.is_eq_cons)
    p = sum(prob.is_eq_cons)
    print(prob.name)
    constraint_wrapper = CUTEstALWrapper(prob)
    m = constraint_wrapper.n_ineq
    print(
        f"Dimension is {n}, {m_plus_p} constraints with {m} inequality and {p} equality")

    #------------------------------------------------
    # Convention: x_slack = [x,a,b] with len(x)=n, len(a)=m, len(b)=p
    def f(x_slack):
        x_temp = x_slack[0:n]
        return prob.obj(x_temp)

    def f_grad(x_slack):
        x_temp = np.array(x_slack[0:n], dtype=np.float64, order='C', copy=True)
        return np.concatenate((prob.obj(x_temp, gradient=True)[1], np.zeros(m + p)))

    if m < 1:
        # only equality constraints
        g = []
        jac_g = []
    else:
        def g(x_slack):
            x_temp = x_slack[0:n]
            a_temp = x_slack[n:n + m]
            return constraint_wrapper.g(x_temp) + a_temp

        def jac_g_matrix(x_slack):
            x_temp = x_slack[0:n]
            x_part = constraint_wrapper.Jg(x_temp)
            a_part = np.eye(m)
            if p > 0:
                b_part = np.zeros((m, p))
                return np.hstack((x_part, a_part, b_part))
            return np.hstack((x_part, a_part))

        def jac_g_product(x_slack, vec):
            x_temp = x_slack[0:n]
            x_part = constraint_wrapper.JgT_v(x_temp, vec)
            b_part = np.zeros(p)

            return np.concatenate((x_part, vec, b_part))

        jac_g = jacobian.Jacobian(jac=jac_g_matrix, jac_prod=jac_g_product)

    if p < 1:
        # only inequality constraints
        h = []
        jac_h = []
    else:
        def h(x_slack):
            x_temp = x_slack[0:n]
            b_temp = x_slack[n + m:n + m + p]
            return constraint_wrapper.h(x_temp) + b_temp

        def jac_h_matrix(x_slack):
            x_temp = x_slack[0:n]
            x_part = constraint_wrapper.Jh(x_temp)
            b_part = np.eye(p)
            if m > 0:
                a_part = np.zeros((p, m))
                return np.hstack((x_part, a_part, b_part))
            return np.hstack((x_part, b_part))

        def jac_h_product(x_slack, vec):
            x_temp = x_slack[0:n]
            x_part = constraint_wrapper.JhT_v(x_temp, vec)
            a_part = np.zeros(m)
            return np.concatenate((x_part, a_part, vec))

        jac_h = jacobian.Jacobian(jac=jac_h_matrix, jac_prod=jac_h_product)

    adapter = IPOPTAdapter(prob)
    problem_ipopt = cyipopt.Problem(
        n=n,
        m=prob.m,
        problem_obj=adapter,
        lb=prob.bl,
        ub=prob.bu,
        cl=prob.cl,
        cu=prob.cu
    )
    return {
        "problem_ipopt": problem_ipopt,
        "prob": prob,
        "n": n,
        "m": m,
        "p": p,
        "f": f,
        "f_grad": f_grad,
        "g": g,
        "jac_g": jac_g,
        "h": h,
        "jac_h": jac_h,
        "constraint_wrapper": constraint_wrapper
    }



def calculate_multiplier(info):
    """
    Calculate the regularization parameter for SALM from the IPOPT multipliers

    Parameters
    ----------
    info : dict
        information about the IPOPT solution

    Returns
    -------
    regularization : float
        regularization parameter

    """
    all_multipliers = info['mult_g']
    max_abs_multiplier = np.max(np.abs(all_multipliers))
    regularization = 10.0 + max_abs_multiplier
    return regularization


def run_cutest_problem(problem_data):
    """
    Runs IPOPT and SALM on a CUTEst problem and writes pickle files

    Parameters
    ----------
    problem_data : dict
        contains problem data

    Returns
    -------
    result : OptimizationResult
        optimization result of SALM of type OptimizationResult

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

    ipopt_x, info = problem_ipopt.solve(prob.x0)
    #print("Solution (IPOPT):\n", ipopt_x)
    print("Objective (IPOPT):\n", prob.obj(ipopt_x))

    with open(f"CUTEst/{problem_class}/{prob.name}_IPOPT_result.pkl", "wb") as write_file:
        pickle.dump(info, write_file)
    regularization = calculate_multiplier(info)
    #print(f"Lower bound: {prob.bl}, upper bound: {prob.bu}")
    phi = L1Slack.L1Slack(n=n, m=m, p=p, sigma=regularization, bl=prob.bl, bu=prob.bu)

    # initial values for SALM
    a0 = -constraint_wrapper.g(prob.x0)
    b0 = -constraint_wrapper.h(prob.x0)
    x0 = np.concatenate((prob.x0, a0, b0), axis=0)


    # Call the solver
    result = SALM.salm(f=f, phi=phi, x0=x0, g=g, h=h, f_grad=f_grad, jac_g=jac_g, jac_h=jac_h, tol=1e-6,
                       maxiter=100, subsolver="PANOC_plus")
    result_x = result.x[0:n]
    print(result.message)
    #print("Optimized solution (SALM):\n", result_x)
    print("Objective (SALM):\n", prob.obj(result_x))
    print("Regularization:\n", phi(result.x))

    with open(f"CUTEst/{problem_class}/{prob.name}_SALM_result.pkl", "wb") as write_file:
        pickle.dump(result, write_file)
    return result


def max_constraint_violation(problem, x):
    """
    Computes maximal constraint violation directly from CUTEst bounds.

    Parameters
    ----------
    problem : pycutest.CUTEstProblem
        contains problem data
    x : np.ndarray
        x-value

    Returns
    -------
    infinity-norm constraint violation
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
    """
    Reads optimization data from folder and calculates metrics for evaluation

    Parameters
    ----------
    subfolder : str
        folder name containing the pickle files of a problem class

    Returns
    -------
    pd.DataFrame with relevant metrics
    """
    folder = Path("CUTEst") / subfolder

    unique_prefixes = {
        file.stem.split("_", 1)[0]
        for file in folder.glob("*.pkl")
    }

    unique_prefixes = sorted(unique_prefixes)
    results = []
    for problem_name in unique_prefixes:
        cutest_prob = pycutest.import_problem(problem_name)

        problem_path = folder / f"{problem_name}_IPOPT_result.pkl"
        with open(problem_path, "rb") as f_ipopt:
            info = pickle.load(f_ipopt)
            x_opt = info["x"]
            constr_violation = max_constraint_violation(cutest_prob, x_opt)
            results.append(
                {
                    "problem_name": problem_name,
                    "method_name": "IPOPT",
                    "obj": info["obj_val"],
                    "constr_violation": constr_violation,
                    "norm_a_plus_b": 0.0,
                    "status": info["status"],
                    "status_msg": info["status_msg"],
                    "outer_iterations": 0,
                    "inner_iterations": 0
                }
            )
        problem_path = folder / f"{problem_name}_SALM_result.pkl"
        with open(problem_path, "rb") as f_salm:
            res_salm = pickle.load(f_salm)
            n = cutest_prob.n
            m = cutest_prob.m
            x_opt = res_salm.x[0:n]
            a_and_b = res_salm.x[n:n + m]
            norm_a_plus_b = np.linalg.norm(a_and_b, ord=np.inf)
            constr_violation = max_constraint_violation(cutest_prob, x_opt)
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




def performance_statistics(df_in: pd.DataFrame):
    """
    Reads result data of a problem class and returns its performance statistics

    Parameters
    ----------
    df_in : pd.DataFrame
        dataframe containing optimization results for a problem class

    Returns
    -------
    pd.DataFrame with statistics
    """
    feasible_ipopt = 0
    feasible_salm = 0
    fbo_ipopt = 0
    fbo_salm = 0
    performs_better_ipopt = 0
    performs_better_salm = 0
    a_small = 0
    kkt_ipopt = 0
    kkt_salm = 0

    cons_tolerance = 1e-5
    a_tolerance = 1e-3
    diff_tolerance = 1e-5

    problem_names = df_in["problem_name"].unique()
    # iterate through unique problem names
    for name in problem_names:
        problem_rows = df_in[df_in["problem_name"] == name]
        ipopt_feas = False
        salm_feas = False
        ipopt_obj = 1e20
        salm_obj = 1e20
        # iterate through solvers
        for index, current_row in problem_rows.iterrows():

            method_name = current_row["method_name"]

            if method_name == "IPOPT":
                ipopt_obj = current_row["obj"]
                if current_row["constr_violation"] < cons_tolerance:
                    feasible_ipopt += 1
                    ipopt_feas = True
                if current_row["status"] >= 0:
                    kkt_ipopt += 1

            if method_name == "SALM":
                salm_obj = current_row["obj"]
                if current_row["constr_violation"] < cons_tolerance:
                    feasible_salm += 1
                    salm_feas = True
                if current_row["norm_a_plus_b"] < a_tolerance:
                    a_small += 1
                if current_row["status"] == 0:
                    kkt_salm += 1

        if ipopt_feas and not salm_feas:
            performs_better_ipopt += 1
        elif salm_feas and not ipopt_feas:
            performs_better_salm += 1

        if ipopt_feas and salm_feas:
            f_diff = (ipopt_obj - salm_obj) / (np.maximum(1.0, np.abs(np.minimum(ipopt_obj, salm_obj))))
            if f_diff >= diff_tolerance:
                fbo_ipopt += 1
                performs_better_ipopt += 1
            elif f_diff <= -diff_tolerance:
                fbo_salm += 1
                performs_better_salm += 1

    ipopt_dict = {
        "method_name": "IPOPT",
        "feasible": feasible_ipopt,
        "feasible, better objective": fbo_ipopt,
        "performs better": performs_better_ipopt,
        "a small": 0,
        "KKT found": kkt_ipopt
    }
    salm_dict = {
        "method_name": "SALM",
        "feasible": feasible_salm,
        "feasible, better objective": fbo_salm,
        "performs better": performs_better_salm,
        "a small": a_small,
        "KKT found": kkt_salm
    }
    return pd.DataFrame([ipopt_dict, salm_dict])


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



problem_class = "sum_of_squares"

#run_problem_set(quadratic_probs)

df = read_data(problem_class)
stat = performance_statistics(df)
#stat.to_csv(f"csv/statistics/{problem_class}_stats.csv", index=False)
#df.to_csv(f"csv/{problem_class}.csv", index=False)

