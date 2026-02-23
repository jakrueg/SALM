from sklearn.externals.array_api_compat.cupy.linalg import outer

from results import OptimizationResult
import numpy as np
from gurobipy import GRB

def build_result(model):

    success = model.status == GRB.OPTIMAL

    x_vars = model.getVars()
    x_opt = np.array([v.X for v in x_vars])
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
        message = model.Message,
        x=x_opt,
        obj=model.Obj,
        nnz=nnz,
        outer_iterations=outer_iterations,
        inner_iterations=inner_iterations,
        ineq_violation=ineq_violation,
        eq_violation=eq_violation
    )