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

Implementations for proximal operators can be found in ``ProxOperators`` or using [pyproximal](https://github.com/PyLops/pyproximal). Subproblem solvers are ``PANOCplus`` adapted from [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl) and a proximal Gradient method from [copt](https://github.com/openopt/copt). As a testing example, we implemented a sparse portfolio optimization problem using data from [Frangioni et al.](https://commalab.di.unipi.it/datasets/mv/)



## Bug reports and support

Please report any issues via the [Github issue tracker](https://github.com/jakrueg/SALM/issues). All types of issues are welcome including bug reports, typos, feature requests and so on.