# This is a sample Python script.
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import results
import portfolio


def plot_performance(metric):

    df = results.load_portfolio("proxGrad")
    df_proxGrad = df[df["success"]==True]
    values_proxGrad = df_proxGrad[metric].values
    values_sorted_proxGrad = np.sort(values_proxGrad)

    num_probs = len(df)



    df = results.load_portfolio("PANOC")
    df_PANOC = df[df["success"]==True]
    values_PANOC = df_PANOC[metric].values
    values_sorted_PANOC = np.sort(values_PANOC)

    x = np.arange(1, np.maximum(np.max(values_sorted_proxGrad),np.max(values_sorted_PANOC)) + 100)
    y_proxGrad = np.copy(x)
    y_PANOC = np.copy(x)
    for i in range(len(x)):
        y_proxGrad[i] = np.sum(values_sorted_proxGrad <= x[i])
        y_PANOC[i] = np.sum(values_sorted_PANOC <= x[i])

    plt.step(x, y_proxGrad,'r',label='Proximal Gradient')
    plt.step(x, y_PANOC,'b',label='PANOC+')
    plt.xlabel(metric)
    plt.ylabel("Number of problems solved")
    plt.ylim(ymin=0, ymax=num_probs)
    #plt.title("Empirical CDF")
    plt.grid(True)
    plt.show()


def generate_data():
    path = Path("/home/jannis/Documents/MV/size200")
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



solvers = ["PANOC","proxGrad"]

portfolio.eval_all_probs(solvers=solvers,write_files=True)



if __name__ == '__main__':
    prob_name = "orl200-05-e"
    #prob_name = "pard200_c"

    #df = generate_data()
    #df.to_csv(f"csv/portfolio/portfolio.csv", index = False)

    res = portfolio.run_problem(problem_name=prob_name,solvers=solvers,write_files=False)
    print(res)

    #solver = "proxGrad"
    #opt_res = res["PANOC"]
    #opt_res = results.load_single_portfolio(prob_name=prob_name,solver=solver)
    #print(opt_res)
    #opt_res2 = res["proxGrad"]
    #print(opt_res2)

    #solver = "proxGrad"
    #df = results.load_portfolio(solver=solver)
    #print(df)
    #solver = "PANOC"
    #df2 = results.load_portfolio(solver=solver)
    #print(df2)
    #solver = "Gurobi"
    #df3 = results.load_portfolio(solver=solver)
    #print(df3)

    #plot_performance("outer_iterations")
    #plot_performance("inner_iterations")

