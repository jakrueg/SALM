import argparse
import csv

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
	parser.add_argument("--maxiter", type=int, default=100)
	parser.add_argument("--tol", type=float, default=1e-3)
	parser.add_argument(
		"--diagnostics",
		action="store_true",
		help="print one inner/outer convergence summary per SALM iteration",
	)
	parser.add_argument(
		"--diagnostics-file",
		help="write detailed SALM iteration diagnostics to this CSV file",
	)
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

		diagnostic_rows = []

		def record_diagnostics(row):
			row = dict(row)
			x_diagnostic = row.pop("x")
			row.update(problem.constraint_diagnostics(x_diagnostic))
			diagnostic_rows.append(row)
			if args.diagnostics:
				print(
					"iter={outer_iteration} pen={penalty:.3g} "
					"inner={inner_success}/{inner_iterations} "
					"res={inner_residual:.3g} "
					"grad={inner_initial_gradient_inf:.3g}->"
					"{inner_final_gradient_inf:.3g} "
					"tau={inner_step_size:.3g} "
					"bt={inner_backtracking_steps}/{inner_max_backtracking_steps} "
					"eq={power_balance_inf:.3g} "
					"bounds={:.3g} network={:.3g}".format(
						row["voltage_bounds_max_violation"]
						+ row["generator_bounds_max_violation"],
						max(
							row["branch_angle_max_violation"],
							row["branch_thermal_max_violation"],
						),
						**row,
					)
				)

		result = SALM.salm(
			f=problem.objective,
			phi=acopf.ZeroPenalty(),
			x0=problem.x0,
			g=problem.salm_inequalities,
			h=problem.salm_equalities,
			f_grad=problem.objective_gradient,
			jac_g=problem.jac_g,
			jac_h=problem.jac_h,
			#subsolver="PANOC_plus",
			maxiter=args.maxiter,
			tol=args.tol,
			diagnostics_callback=(
				record_diagnostics
				if args.diagnostics or args.diagnostics_file else None
			),
		)
		x = result.x
		status = result.status
		print("Solver: SALM")
		print("Status:", result.message)
		print("Outer iterations:", result.outer_iterations)
		print("Inner iterations:", result.inner_iterations)
		if args.diagnostics_file:
			fieldnames = [
				key for key in diagnostic_rows[0] if key != "x"
			] if diagnostic_rows else []
			with open(args.diagnostics_file, "w", newline="", encoding="utf-8") as file:
				writer = csv.DictWriter(file, fieldnames=fieldnames)
				if fieldnames:
					writer.writeheader()
					writer.writerows(diagnostic_rows)
			print("Diagnostics CSV:", args.diagnostics_file)

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
