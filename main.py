
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import results
import portfolio

def compress_profile(tau, rho):
    """
    Compresses a dense performance profile in one, which only saves the points where the rho value changes

    Parameters
    ----------
    tau : array_like
        input space, number of iterations
    rho : array_like
        function values, number of problems solved

    Returns
    -------
    tau, rho
        vectors with compressed data
    """
    tau = np.asarray(tau)
    rho = np.asarray(rho)

    keep = [0]

    for i in range(1, len(rho)):
        if rho[i] != rho[i-1]:   # step change
            keep.append(i)

    keep.append(len(rho)-1)

    return tau[keep], rho[keep]

def plot_performance(metric):
    """
    Reads result data from portfolio optimization problems and builds a performance profile, which is shown
    and exported as .csv

    Parameters
    ----------
    metric : str
        metric for which the performance profile is built

    """
    # load PG results
    df = results.load_portfolio("proxGrad")
    df_proxGrad = df[df["success"]==True]
    values_proxGrad = df_proxGrad[metric].values
    values_sorted_proxGrad = np.sort(values_proxGrad)

    num_probs = len(df)


    # load PANOC+ results
    df = results.load_portfolio("PANOC")
    df_PANOC = df[df["success"]==True]
    values_PANOC = df_PANOC[metric].values
    values_sorted_PANOC = np.sort(values_PANOC)

    # performance profile
    x = np.arange(1, np.maximum(np.max(values_sorted_proxGrad),np.max(values_sorted_PANOC)) + 10)
    y_proxGrad = np.copy(x)
    y_PANOC = np.copy(x)
    for i in range(len(x)):
        y_proxGrad[i] = np.sum(values_sorted_proxGrad <= x[i])
        y_PANOC[i] = np.sum(values_sorted_PANOC <= x[i])

    #export plot
    tau_pg , rho_pg = compress_profile(x, y_proxGrad)
    df_pg = pd.DataFrame(
        {
            "tau": tau_pg,
            "rho": rho_pg
        }
    )
    df_pg.to_csv(f"ppPG{metric}.csv", index=False)

    tau_panoc, rho_panoc = compress_profile(x, y_PANOC)
    dfPANOC = pd.DataFrame(
        {
            "tau": tau_panoc,
            "rho": rho_panoc
        }
    )
    dfPANOC.to_csv(f"ppPANOC{metric}.csv", index=False)

    # show plot
    plt.step(x, y_proxGrad,'r',label='Proximal Gradient')
    plt.step(x, y_PANOC,'b',label='PANOC+')
    plt.xlabel(metric)
    plt.ylabel("Number of problems solved")
    plt.ylim(ymin=0, ymax=num_probs)
    plt.grid(True)
    plt.show()


def generate_data():
    """
    Generates a single DataFrame from the optimization results of all single portfolio problems


    Returns
    -------
    pd.DataFrame with optimization results of all portfolio problems
    """
    path = Path("/home/krueger/MV/size200")
    problem_files = sorted(path.glob("*.mat"))
    results = []
    for problem_file in problem_files:
        problem_name = problem_file.stem

        Q = portfolio.read_covariance_matrix(path,problem_name)
        mu = portfolio.read_mu(path,problem_name)
        rho = portfolio.read_rho(path,problem_name)

        with open(f"Data/{problem_name}_Gurobi_result.pkl", "rb") as f_gurobi:
            res_gurobi = pickle.load(f_gurobi)
            obj_reference = res_gurobi.obj
            x_gurobi = res_gurobi.x
            results.append(
                {
                    "problem_name": problem_name,
                    "method_name" : "Gurobi",
                    "outer_iterations": int(res_gurobi.outer_iterations),
                    "inner_iterations": int(res_gurobi.inner_iterations),
                    "obj" : obj_reference,
                    "nnz": res_gurobi.nnz,
                    "relative_error": 0.0,
                    "risk" : 0.5*x_gurobi.T @ Q @ x_gurobi,
                    "return" : np.dot(mu, x_gurobi),
                    "budget" : np.sum(x_gurobi),
                }
            )

        with open(f"Data/{problem_name}_proxGrad_result.pkl", "rb") as f_proxGrad:
            res_proxGrad = pickle.load(f_proxGrad)
            obj_proxGrad = res_proxGrad.obj
            relative_error_pg = (obj_proxGrad - obj_reference)/obj_reference
            x_proxGrad = res_proxGrad.x
            results.append(
                {
                    "problem_name": problem_name,
                    "method_name": "SALM(PG)",
                    "outer_iterations": int(res_proxGrad.outer_iterations),
                    "inner_iterations": int(res_proxGrad.inner_iterations),
                    "obj": obj_proxGrad,
                    "nnz": res_proxGrad.nnz,
                    "relative_error": relative_error_pg,
                    "risk": 0.5 * x_proxGrad.T @ Q @ x_proxGrad,
                    "return": np.dot(mu, x_proxGrad),
                    "budget": np.sum(x_proxGrad),
                }
            )

        with open(f"Data/{problem_name}_PANOC_result.pkl", "rb") as f_panoc:
            res_panoc = pickle.load(f_panoc)
            obj_panoc = res_panoc.obj
            relative_error = (obj_panoc - obj_reference) / obj_reference
            x_panoc = res_panoc.x
            results.append(
                {
                    "problem_name": problem_name,
                    "method_name": "SALM(PANOC+)",
                    "outer_iterations": int(res_panoc.outer_iterations),
                    "inner_iterations": int(res_panoc.inner_iterations),
                    "obj": obj_panoc,
                    "nnz": res_panoc.nnz,
                    "relative_error": relative_error,
                    "risk": 0.5 * x_panoc.T @ Q @ x_panoc,
                    "return": np.dot(mu, x_panoc),
                    "budget": np.sum(x_panoc),
                }
            )
    return pd.DataFrame(results)



solvers = ["PANOC","proxGrad", "Gurobi"]
#solvers = ["proxGrad"]
portfolio.eval_all_probs(solvers=solvers,write_files=True)



if __name__ == '__main__':
    prob_name = "orl200-05-d"
    #prob_name = "pard200_d"

    df = generate_data()
    df.to_csv(f"csv/portfolio/portfolio.csv", index = False)

    #res = portfolio.run_problem(problem_name=prob_name,solvers=solvers,write_files=False)
    #print(res)

    #solver = "proxGrad"
    #opt_res = res["PANOC"]
    #opt_res = results.load_single_portfolio(prob_name=prob_name,solver=solver)
    #print(opt_res)
    #opt_res2 = res["proxGrad"]
    #print(opt_res2)
    #df = generate_data()
    #metric = "inner_iterations"

    #df_pg = df[df["method_name"] == "SALM(PG)"]
    #med_pg = df_pg[metric].median()
    #df_panoc = df[df["method_name"] == "SALM(PANOC+)"]
    #med_panoc = df_panoc[metric].median()
    #print(f"Median PG: {med_pg}, Median PANOC: {med_panoc}")
    #print(df2)
    #solver = "Gurobi"
    #df3 = results.load_portfolio(solver=solver)
    #print(df3)

    #plot_performance("outer_iterations")
    #plot_performance("inner_iterations")

