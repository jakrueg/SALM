
from results import OptimizationResult
import numpy as np
from gurobipy import GRB

def build_result(model,x):

    success = model.status == GRB.OPTIMAL

    x_opt = x
    print(x_opt)
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
        obj=model.ObjVal,
        nnz=nnz,
        outer_iterations=outer_iterations,
        inner_iterations=inner_iterations,
        ineq_violation=ineq_violation,
        eq_violation=eq_violation
    )