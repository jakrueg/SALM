# SALM

Python Implementation of the Safeguarded Augmented Lagrangian Method.

## Synopsis
This package contains an implementation of the Safeguarded Augmented Lagrangian Method (SALM) for constrained composite optimization problems of the following form.
```
   minimize        f(x) + phi(x)
   over            x ∈ Rⁿ
   subject to      g(x)≤0, h(x)=0
```
where ``f``, ``g`` and ``h`` are continuously differentiable and ``phi`` is locally Lipschitz continuous.
``phi`` should be prox-friendly.

The problem terms are accessed through the following oracles:

* ```f```: function value f(x) and gradient ∇f(x)
* ```phi```: proximal mapping prox_phi(x) and function value at proximal point phi(z)
* ```g```: function value g(x) and Jacobian ∇g(x)
* ```h```: function value h(x) and Jacobian ∇h(x)

Implementations for proximal operators can be found in ``ProxOperators`` or using [pyproximal](https://github.com/PyLops/pyproximal). Subproblem solvers are ``PANOCplus`` adapted from [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl) and a proximal Gradient method from [this paper](https://doi.org/10.1007/s10957-022-02101-3). As a testing example, we implemented a sparse portfolio optimization problem using data from [Frangioni et al.](https://commalab.di.unipi.it/datasets/mv/)

## AC optimal power flow

``acopf.py`` loads MATPOWER/PGLib cases and provides the AC-OPF cost, analytic
gradient, power-balance equations, and sparse constraint Jacobians. For SALM,
use ``salm_inequalities`` (the solver's convention is ``g(x) <= 0``) and
``salm_equalities`` (power balance plus the reference-bus angle). The
``jac_g`` and ``jac_h`` attributes are SALM Jacobian wrappers with sparse
Jacobian-vector products;
``inequality_jacobian`` and ``equality_jacobian`` also expose the sparse
matrices directly.

```python
from Algorithms.SALM import salm
from acopf import ZeroPenalty, load_problem

problem = load_problem("pglib-opf-master", "pglib_opf_case14_ieee")
result = salm(
    f=problem.objective,
    f_grad=problem.objective_gradient,
    phi=ZeroPenalty(),
    x0=problem.x0,
    g=problem.salm_inequalities,
    jac_g=problem.jac_g,
    h=problem.salm_equalities,
    jac_h=problem.jac_h,
)
```

MATPOWER piecewise-linear generation costs (model 1) have kinks; their
``objective_gradient`` uses the slope of the active segment. For such cases,
the objective is not continuously differentiable at breakpoints, so SALM's
smooth-objective assumptions do not strictly apply there.

For an IPOPT reference solution, install ``cyipopt`` and IPOPT, then run:

```bash
python acopf-runner.py --solver ipopt --case pglib_opf_case14_ieee
```

IPOPT is the runner's default; pass ``--solver salm`` to run SALM instead.
Variable limits are supplied to IPOPT as native box bounds, while power
balances and branch-flow/angle limits are supplied as nonlinear constraints.

For SALM diagnostics, pass ``--diagnostics`` to print a compact summary per
outer iteration, or ``--diagnostics-file salm-diagnostics.csv`` to save detailed
inner/outer metrics and AC-OPF constraint-group violations as CSV. The
proximal-gradient residual is the norm of the proximal-gradient mapping,
``||x - prox(x - tau * grad(F(x)), tau)|| / tau``.



## Bug reports and support

Please report any issues via the [Github issue tracker](https://github.com/jakrueg/SALM/issues). All types of issues are welcome including bug reports, typos, feature requests and so on.
