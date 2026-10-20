import argparse

import numpy as np

import acopf


def main():
	parser = argparse.ArgumentParser(description="Run a PGLib AC-OPF case.")
	parser.add_argument(
		"--solver",
		choices=("ipopt", "salm"),
		default="salm",
		help="solver to use (IPOPT is the reference default)",
	)
	parser.add_argument(
		"--case",
		default="pglib_opf_case14_ieee",
		help="PGLib case filename or stem",
	)
	parser.add_argument("--maxiter", type=int, default=2000)
	parser.add_argument("--tol", type=float, default=1e-8)
	args = parser.parse_args()

	problem = acopf.load_problem("pglib-opf-master", args.case)
	if args.solver == "ipopt":
		x, info = problem.solve_ipopt({
			"max_iter": args.maxiter,
			"tol": args.tol,
		})
		status = info["status"]
		print("Solver: IPOPT")
		status_message = info["status_msg"]
		if isinstance(status_message, bytes):
			status_message = status_message.decode(errors="replace")
		print("Status:", status_message)
	else:
		from Algorithms import SALM

		result = SALM.salm(
			f=problem.objective,
			phi=acopf.ZeroPenalty(),
			x0=problem.x0,
			g=problem.salm_inequalities,
			h=problem.salm_equalities,
			f_grad=problem.objective_gradient,
			jac_g=problem.jac_g,
			jac_h=problem.jac_h,
			subsolver="PANOC_plus",
			maxiter=args.maxiter,
			tol=args.tol,
		)
		x = result.x
		status = result.status
		print("Solver: SALM")
		print("Status:", result.message)
		print("Outer iterations:", result.outer_iterations)
		print("Inner iterations:", result.inner_iterations)

	print("Objective:", problem.objective(x))
	print("Power-balance equality violation:",
		  np.linalg.norm(problem.equalities(x), ord=np.inf))
	print("Network-limit violation:",
		  np.max(np.maximum(-problem.network_inequalities(x), 0.0), initial=0.0))
	bound_violations = [
		max(lower - value, 0.0) if lower is not None else 0.0
		for value, (lower, _) in zip(x, problem.bounds)
	] + [
		max(value - upper, 0.0) if upper is not None else 0.0
		for value, (_, upper) in zip(x, problem.bounds)
	]
	print("Variable-bound violation:", max(bound_violations, default=0.0))
	return 0 if status in (0, 1) else 1


if __name__ == "__main__":
	raise SystemExit(main())
