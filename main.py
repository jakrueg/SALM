# This is a sample Python script.
import numpy as np
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





solvers = ["PANOC","proxGrad"]

#portfolio.eval_all_probs(solvers=solvers,write_files=True)



if __name__ == '__main__':
    prob_name = "orl200-005-h"
    #prob_name = "pard200_b"


    solver = "proxGrad"
    res = portfolio.run_problem(problem_name=prob_name,solvers=solvers,write_files=False)
    #print(res)
    opt_res = res["PANOC"]
    #opt_res = results.load_single_portfolio(prob_name=prob_name,solver=solver)
    print(opt_res)
    opt_res2 = res["proxGrad"]
    print(opt_res2)
    df = results.load_portfolio(solver=solver)
    print(df)
    solver = "PANOC"
    df2 = results.load_portfolio(solver=solver)
    print(df2)
    plot_performance("outer_iterations")
    plot_performance("inner_iterations")

