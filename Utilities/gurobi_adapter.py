
from results import OptimizationResult
import numpy as np
from gurobipy import GRB

def build_result(model,x, Sigma, reg):
    r"""
        Builds an optimization result out of the data provided by the Gurobi model.

        Parameters
        ----------
        model : gurobipy.Model
            Gurobi model
        x : np.array
            optimal vector

        Returns
        -------
        OptimizationResult
            optimization result of type OptimizationResult
    """


    success = model.status == GRB.OPTIMAL

    x_opt = x
    #print(x_opt)
    nnz = int(np.count_nonzero(x_opt))

    inner_iterations = model.IterCount
    outer_iterations = inner_iterations

    eq_violation = 0.0
    ineq_violation = 0.0

    for constr in model.getConstrs():
        slack = constr.Slack
        if constr.Sense == '=':
            eq_violation = max(eq_violation, abs(slack))
        else:
            ineq_violation = max(ineq_violation, max(0, -slack))

    return OptimizationResult(
        success=success,
        status= model.Status,
        message = f"Model built with gurobi, finished with status {model.Status}",
        x=x_opt,
        obj=0.5 * x_opt @ Sigma @ x_opt+ reg * nnz,
        nnz=nnz,
        outer_iterations=int(outer_iterations),
        inner_iterations=int(inner_iterations),
        ineq_violation=ineq_violation,
        eq_violation=eq_violation
    )