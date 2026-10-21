"""Load MATPOWER/PGLib AC-OPF cases as objective and constraint functions.

The decision vector is ``[Va (rad), Vm (p.u.), Pg (MW), Qg (MVAr)]``.
``inequalities`` returns positive feasibility margins; ``salm_inequalities``
converts them to SALM's ``g(x) <= 0`` convention. The problem exposes analytic
derivatives and sparse Jacobians for the objective and constraints.
"""

from pathlib import Path
import re

import numpy as np
from scipy.sparse import coo_matrix, vstack

from Utilities.jacobian import Jacobian


class ZeroPenalty:
	"""Zero-valued nonsmooth term with the identity proximal operator."""

	def __call__(self, x):
		return 0.0

	def prox(self, x, tau):
		return np.asarray(x, dtype=float).copy()


# Zero-based MATPOWER matrix column indices.
BUS_I, BUS_TYPE, PD, QD, GS, BS, VM, VA, VMAX, VMIN = 0, 1, 2, 3, 4, 5, 7, 8, 11, 12
GEN_BUS, PG, QG, QMAX, QMIN, GEN_STATUS, PMAX, PMIN = 0, 1, 2, 3, 4, 7, 8, 9
F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX = 0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12


def _read_matpower(path):
	"""Read the numeric fields used by standard MATPOWER/PGLib case files."""
	text = Path(path).read_text(encoding="utf-8", errors="replace")
	text = re.sub(r"%[^\n]*", "", text).replace("...", " ")
	fields = {}

	# Matrices must be parsed separately because their rows also end in ';'.
	for name, body in re.findall(r"mpc\.(\w+)\s*=\s*\[(.*?)\]\s*;", text, re.S):
		rows = []
		for row in body.split(";"):
			row = row.strip()
			if row:
				rows.append([float(value.replace("D", "E").replace("d", "e"))
							 for value in row.replace(",", " ").split()])
		fields[name] = np.asarray(rows, dtype=float)

	for name, value in re.findall(r"mpc\.(\w+)\s*=\s*([^;]+);", text):
		try:
			fields[name] = float(value.strip())
		except ValueError:
			pass
	missing = [key for key in ("baseMVA", "bus", "gen", "branch", "gencost")
			   if key not in fields]
	if missing:
		raise ValueError(f"{path} is missing MATPOWER fields: {', '.join(missing)}")
	return fields


class ACOPFProblem:
	"""AC-OPF data and callable functions suitable for an AL optimizer."""

	def __init__(self, data, name=""):
		self.name = name
		self.base_mva = float(data["baseMVA"])
		self.bus = data["bus"]
		self.gen = data["gen"]
		self.branch = data["branch"]
		self.gencost = data["gencost"]
		self.n_bus, self.n_gen = len(self.bus), len(self.gen)
		self._bus_index = {int(row[BUS_I]): i for i, row in enumerate(self.bus)}
		self._reference_buses = np.flatnonzero(self.bus[:, BUS_TYPE] == 3)
		if len(self._bus_index) != self.n_bus:
			raise ValueError("MATPOWER bus numbers must be unique")
		if len(self.gencost) < self.n_gen:
			raise ValueError("gencost must contain one row per generator")
		self._build_admittance()
		self.x0, self.bounds = self._make_initial_point_and_bounds()
		self.jac_g = Jacobian(
			jac=self.salm_inequality_jacobian,
			jac_prod=self._salm_inequality_jacobian_product,
		)
		self.jac_h = Jacobian(
			jac=self.salm_equality_jacobian,
			jac_prod=self._salm_equality_jacobian_product,
		)

	@property
	def dimension(self):
		return 2 * self.n_bus + 2 * self.n_gen

	def _build_admittance(self):
		ybus_rows = list(range(self.n_bus))
		ybus_columns = list(range(self.n_bus))
		ybus_values = [
			(bus[GS] + 1j * bus[BS]) / self.base_mva for bus in self.bus
		]
		self._branch_y = []
		for row in self.branch:
			f, t = self._bus_index[int(row[F_BUS])], self._bus_index[int(row[T_BUS])]
			if row[BR_STATUS] == 0:
				self._branch_y.append((f, t, 0j, 0j, 0j, 0j))
				continue
			z = row[BR_R] + 1j * row[BR_X]
			y = 1 / z if z != 0 else 0j
			tap = row[TAP] if row[TAP] != 0 else 1.0
			tap *= np.exp(1j * np.deg2rad(row[SHIFT]))
			yff = (y + 1j * row[BR_B] / 2) / abs(tap) ** 2
			yft, ytf = -y / np.conj(tap), -y / tap
			ytt = y + 1j * row[BR_B] / 2
			ybus_rows.extend((f, f, t, t))
			ybus_columns.extend((f, t, f, t))
			ybus_values.extend((yff, yft, ytf, ytt))
			self._branch_y.append((f, t, yff, yft, ytf, ytt))
		self.ybus = coo_matrix(
			(ybus_values, (ybus_rows, ybus_columns)),
			shape=(self.n_bus, self.n_bus),
		).tocsr()

	def _make_initial_point_and_bounds(self):
		va = np.deg2rad(self.bus[:, VA])
		x0 = np.concatenate((va, self.bus[:, VM], self.gen[:, PG], self.gen[:, QG]))
		bounds = [(None, None)] * self.dimension
		for i, bus in enumerate(self.bus):
			bounds[self.n_bus + i] = (bus[VMIN], bus[VMAX])
			if bus[BUS_TYPE] == 3:
				bounds[i] = (va[i], va[i])
		p_offset = 2 * self.n_bus
		q_offset = p_offset + self.n_gen
		for i, gen in enumerate(self.gen):
			if gen[GEN_STATUS] == 0:
				bounds[p_offset + i] = bounds[q_offset + i] = (0.0, 0.0)
				x0[p_offset + i] = x0[q_offset + i] = 0.0
			else:
				bounds[p_offset + i] = (gen[PMIN], gen[PMAX])
				bounds[q_offset + i] = (gen[QMIN], gen[QMAX])
		return x0, bounds

	def _unpack(self, x):
		x = np.asarray(x, dtype=float)
		if x.size != self.dimension:
			raise ValueError(f"Expected a decision vector of length {self.dimension}")
		n = self.n_bus
		return x[:n], x[n:2*n], x[2*n:2*n+self.n_gen], x[2*n+self.n_gen:]

	def objective(self, x):
		"""Total active-generation cost, using each case's gencost model."""
		_, _, pg, _ = self._unpack(x)
		total = 0.0
		for i, gen in enumerate(self.gen):
			if gen[GEN_STATUS] == 0:
				continue
			cost = self.gencost[i]
			model, count = int(cost[0]), int(cost[3])
			coefficients = cost[4:4 + count]
			if model == 2:
				total += np.polyval(coefficients, pg[i])
			elif model == 1:
				points = coefficients.reshape(-1, 2)
				total += np.interp(pg[i], points[:, 0], points[:, 1])
			else:
				raise ValueError(f"Unsupported gencost model {model}")
		return float(total)

	def objective_gradient(self, x):
		"""Gradient of the generation cost with respect to the decision vector."""
		_, _, pg, _ = self._unpack(x)
		gradient = np.zeros(self.dimension)
		p_offset = 2 * self.n_bus
		for i, gen in enumerate(self.gen):
			if gen[GEN_STATUS] == 0:
				continue
			cost = self.gencost[i]
			model, count = int(cost[0]), int(cost[3])
			coefficients = cost[4:4 + count]
			if model == 2:
				gradient[p_offset + i] = np.polyval(np.polyder(coefficients), pg[i])
			elif model == 1:
				points = coefficients.reshape(-1, 2)
				if points[0, 0] <= pg[i] <= points[-1, 0]:
					segment = np.searchsorted(points[:, 0], pg[i], side="right") - 1
					segment = np.clip(segment, 0, len(points) - 2)
					gradient[p_offset + i] = (
						(points[segment + 1, 1] - points[segment, 1])
						/ (points[segment + 1, 0] - points[segment, 0])
					)
			else:
				raise ValueError(f"Unsupported gencost model {model}")
		return gradient

	def equalities(self, x):
		"""Nodal active/reactive power balance residuals in MW and MVAr."""
		va, vm, pg, qg = self._unpack(x)
		voltage = vm * np.exp(1j * va)
		network = voltage * np.conj(self.ybus @ voltage) * self.base_mva
		generation_p = np.zeros(self.n_bus)
		generation_q = np.zeros(self.n_bus)
		for i, gen in enumerate(self.gen):
			if gen[GEN_STATUS] != 0:
				bus = self._bus_index[int(gen[GEN_BUS])]
				generation_p[bus] += pg[i]
				generation_q[bus] += qg[i]
		return np.r_[generation_p - self.bus[:, PD] - network.real,
					 generation_q - self.bus[:, QD] - network.imag]

	def equality_jacobian(self, x):
		"""Sparse Jacobian of the nodal active/reactive power balance equations."""
		va, vm, _, _ = self._unpack(x)
		rows, columns, values = [], [], []
		n = self.n_bus
		pg_offset, qg_offset = 2 * n, 2 * n + self.n_gen
		p_angle_diagonal = np.zeros(n)
		q_angle_diagonal = np.zeros(n)
		p_voltage_diagonal = np.zeros(n)
		q_voltage_diagonal = np.zeros(n)

		# Nodal derivatives follow the standard polar AC power-flow equations.
		for i in range(n):
			start, end = self.ybus.indptr[i:i + 2]
			neighbors = zip(self.ybus.indices[start:end], self.ybus.data[start:end])
			y_ii = self.ybus[i, i]
			p_voltage_diagonal[i] = 2 * vm[i] * y_ii.real * self.base_mva
			q_voltage_diagonal[i] = -2 * vm[i] * y_ii.imag * self.base_mva
			for j, y_ij in neighbors:
				if i == j:
					continue
				g_ij, b_ij = y_ij.real, y_ij.imag
				delta = va[i] - va[j]
				cos_delta, sin_delta = np.cos(delta), np.sin(delta)
				g_cos_b_sin = g_ij * cos_delta + b_ij * sin_delta
				g_sin_b_cos = g_ij * sin_delta - b_ij * cos_delta
				p_angle = -self.base_mva * vm[i] * vm[j] * g_sin_b_cos
				q_angle = self.base_mva * vm[i] * vm[j] * g_cos_b_sin
				rows.extend((i, i, n + i, n + i))
				columns.extend((j, n + j, j, n + j))
				values.extend((
					p_angle,
					-self.base_mva * vm[i] * g_cos_b_sin,
					q_angle,
					-self.base_mva * vm[i] * g_sin_b_cos,
				))
				p_angle_diagonal[i] -= p_angle
				q_angle_diagonal[i] -= q_angle
				p_voltage_diagonal[i] += self.base_mva * vm[j] * g_cos_b_sin
				q_voltage_diagonal[i] += self.base_mva * vm[j] * g_sin_b_cos

		for i in range(n):
			rows.extend((i, i, n + i, n + i))
			columns.extend((i, n + i, i, n + i))
			values.extend((
				p_angle_diagonal[i],
				-p_voltage_diagonal[i],
				q_angle_diagonal[i],
				-q_voltage_diagonal[i],
			))

		# Generation enters the matching bus balance with coefficient +1.
		for i, gen in enumerate(self.gen):
			if gen[GEN_STATUS] == 0:
				continue
			bus = self._bus_index[int(gen[GEN_BUS])]
			rows.extend((bus, n + bus))
			columns.extend((pg_offset + i, qg_offset + i))
			values.extend((1.0, 1.0))
		return coo_matrix(
			(values, (rows, columns)), shape=(2 * n, self.dimension)
		).tocsr()

	def salm_equalities(self, x):
		"""Power balances, reference angles and offline-generator fixes for SALM."""
		va, _, pg, qg = self._unpack(x)
		offline_generators = np.flatnonzero(self.gen[:, GEN_STATUS] == 0)
		return np.r_[
			self.equalities(x),
			va[self._reference_buses]
			- np.deg2rad(self.bus[self._reference_buses, VA]),
			pg[offline_generators],
			qg[offline_generators],
		]

	def salm_equality_jacobian(self, x):
		"""Sparse Jacobian of salm_equalities()."""
		offline_generators = np.flatnonzero(self.gen[:, GEN_STATUS] == 0)
		n_reference = len(self._reference_buses)
		n_offline = len(offline_generators)
		rows = np.r_[
			np.arange(n_reference),
			n_reference + np.arange(n_offline),
			n_reference + n_offline + np.arange(n_offline),
		]
		columns = np.r_[
			self._reference_buses,
			2 * self.n_bus + offline_generators,
			2 * self.n_bus + self.n_gen + offline_generators,
		]
		reference_jacobian = coo_matrix(
			(
				np.ones(len(rows)),
				(rows, columns),
			),
			shape=(n_reference + 2 * n_offline, self.dimension),
		).tocsr()
		return vstack((self.equality_jacobian(x), reference_jacobian), format="csr")

	def inequalities(self, x):
		"""Voltage, generation, branch-flow and angle margins (all >= 0)."""
		va, vm, pg, qg = self._unpack(x)
		margins = [vm - self.bus[:, VMIN], self.bus[:, VMAX] - vm]
		for i, gen in enumerate(self.gen):
			if gen[GEN_STATUS] != 0:
				margins.extend([np.array([pg[i] - gen[PMIN]]),
								np.array([gen[PMAX] - pg[i]]),
								np.array([qg[i] - gen[QMIN]]),
								np.array([gen[QMAX] - qg[i]])])
		return np.concatenate((np.concatenate(margins), self.network_inequalities(x)))

	def network_inequalities(self, x):
		"""Branch-flow and angle margins, excluding variable-bound constraints."""
		va, vm, _, _ = self._unpack(x)
		voltage = vm * np.exp(1j * va)
		margins = []
		for k, row in enumerate(self.branch):
			f, t, yff, yft, ytf, ytt = self._branch_y[k]
			if row[BR_STATUS] == 0:
				continue
			angle = va[f] - va[t]
			if row[ANGMIN] > -360:
				margins.append(np.array([angle - np.deg2rad(row[ANGMIN])]))
			if row[ANGMAX] < 360:
				margins.append(np.array([np.deg2rad(row[ANGMAX]) - angle]))
			if row[RATE_A] > 0:
				vf, vt = voltage[f], voltage[t]
				sf = vf * np.conj(yff * vf + yft * vt) * self.base_mva
				st = vt * np.conj(ytf * vf + ytt * vt) * self.base_mva
				margins.extend([np.array([row[RATE_A]**2 - abs(sf)**2]), np.array([row[RATE_A]**2 - abs(st)**2])])
		return np.concatenate(margins).astype(float) if margins else np.zeros(0)

	def salm_inequalities(self, x):
		"""SALM inequality vector in its required convention, g(x) <= 0."""
		return -self.inequalities(x)

	def constraint_diagnostics(self, x):
		"""Return physical-unit violation summaries grouped by constraint type."""
		va, vm, _, _ = self._unpack(x)
		margins = self.inequalities(x)
		n, row = self.n_bus, 2 * self.n_bus
		online_generators = int(np.count_nonzero(self.gen[:, GEN_STATUS] != 0))
		generator_end = row + 4 * online_generators

		def summarize(name, values):
			violations = np.maximum(-np.asarray(values), 0.0)
			return {
				f"{name}_max_violation": float(
					np.max(violations, initial=0.0)
				),
				f"{name}_violated_count": int(np.count_nonzero(violations > 1e-6)),
			}

		diagnostics = summarize("voltage_bounds", margins[:row])
		diagnostics.update(summarize("generator_bounds", margins[row:generator_end]))

		angle_margins, flow_margins = [], []
		voltage = vm * np.exp(1j * va)
		for k, branch in enumerate(self.branch):
			f, t, yff, yft, ytf, ytt = self._branch_y[k]
			if branch[BR_STATUS] == 0:
				continue
			angle = va[f] - va[t]
			if branch[ANGMIN] > -360:
				angle_margins.append(angle - np.deg2rad(branch[ANGMIN]))
			if branch[ANGMAX] < 360:
				angle_margins.append(np.deg2rad(branch[ANGMAX]) - angle)
			if branch[RATE_A] > 0:
				vf, vt = voltage[f], voltage[t]
				sf = vf * np.conj(yff * vf + yft * vt) * self.base_mva
				st = vt * np.conj(ytf * vf + ytt * vt) * self.base_mva
				flow_margins.extend((
					branch[RATE_A] ** 2 - abs(sf) ** 2,
					branch[RATE_A] ** 2 - abs(st) ** 2,
				))
		diagnostics.update(summarize("branch_angle", angle_margins))
		diagnostics.update(summarize("branch_thermal", flow_margins))

		equalities = self.salm_equalities(x)
		diagnostics["power_balance_inf"] = float(
			np.linalg.norm(equalities[:2 * n], ord=np.inf)
		)
		diagnostics["fixed_equalities_inf"] = float(
			np.linalg.norm(equalities[2 * n:], ord=np.inf)
			if equalities.size > 2 * n else 0.0
		)
		return diagnostics

	def inequality_jacobian(self, x):
		"""Sparse Jacobian of the positive feasibility margins from inequalities()."""
		rows, columns, values = [], [], []
		n, ng = self.n_bus, self.n_gen

		# Voltage magnitude lower/upper margins.
		for i in range(n):
			rows.extend((i, n + i))
			columns.extend((n + i, n + i))
			values.extend((1.0, -1.0))

		# Generator active/reactive lower/upper margins.
		row = 2 * n
		for i, gen in enumerate(self.gen):
			if gen[GEN_STATUS] == 0:
				continue
			rows.extend((row, row + 1, row + 2, row + 3))
			columns.extend((2 * n + i, 2 * n + i, 2 * n + ng + i, 2 * n + ng + i))
			values.extend((1.0, -1.0, 1.0, -1.0))
			row += 4

		bounds_jacobian = coo_matrix(
			(values, (rows, columns)),
			shape=(row, self.dimension),
		).tocsr()
		return vstack(
			(bounds_jacobian, self.network_inequality_jacobian(x)),
			format="csr",
		)

	def network_inequality_jacobian(self, x):
		"""Sparse Jacobian of branch-flow and angle margins only."""
		va, vm, _, _ = self._unpack(x)
		rows, columns, values = [], [], []
		n = self.n_bus
		row = 0
		voltage = vm * np.exp(1j * va)
		for k, branch in enumerate(self.branch):
			f, t, yff, yft, ytf, ytt = self._branch_y[k]
			if branch[BR_STATUS] == 0:
				continue
			if branch[ANGMIN] > -360:
				rows.extend((row, row))
				columns.extend((f, t))
				values.extend((1.0, -1.0))
				row += 1
			if branch[ANGMAX] < 360:
				rows.extend((row, row))
				columns.extend((f, t))
				values.extend((-1.0, 1.0))
				row += 1
			if branch[RATE_A] > 0:
				for a, b, ya, yab in ((f, t, yff, yft), (t, f, ytt, ytf)):
					voltage_a, voltage_b = voltage[a], voltage[b]
					angle_a, angle_b = va[a], va[b]
					current = ya * voltage_a + yab * voltage_b
					apparent_power = self.base_mva * voltage_a * np.conj(current)
					cross = self.base_mva * voltage_a * np.conj(yab * voltage_b)
					d_s = (
						1j * cross,
						-1j * cross,
						self.base_mva * (
							np.exp(1j * angle_a)
							* np.conj(current)
							+ voltage_a * np.conj(ya * np.exp(1j * angle_a))
						),
						self.base_mva * voltage_a * np.conj(yab * np.exp(1j * angle_b)),
					)
					# The first two entries correspond to angle derivatives at a,b;
					# the last two correspond to magnitude derivatives at a,b.
					local_columns = (a, b, n + a, n + b)
					for column, derivative in zip(local_columns, d_s):
						value = -2.0 * np.real(np.conj(apparent_power) * derivative)
						if value != 0.0:
							rows.append(row)
							columns.append(column)
							values.append(value)
					row += 1
		expected_rows = self.network_inequalities(x).size
		if row != expected_rows:
			raise RuntimeError("Network Jacobian rows do not match the constraint vector")
		return coo_matrix(
			(values, (rows, columns)), shape=(row, self.dimension)
		).tocsr()

	def ipopt_constraints(self, x):
		"""IPOPT constraints: power balance and network limits (all <= 0 except balances)."""
		return np.r_[self.equalities(x), -self.network_inequalities(x)]

	def ipopt_constraint_jacobian(self, x):
		"""Sparse Jacobian matching ipopt_constraints()."""
		return vstack(
			(self.equality_jacobian(x), -self.network_inequality_jacobian(x)),
			format="csr",
		)

	def ipopt_jacobian_structure(self):
		"""Return a stable sparse coordinate pattern for IPOPT's Jacobian callback."""
		rows, columns = [], []
		n = self.n_bus
		for i in range(n):
			start, end = self.ybus.indptr[i:i + 2]
			for j in self.ybus.indices[start:end]:
				rows.extend((i, i, n + i, n + i))
				columns.extend((j, n + j, j, n + j))
		for i, gen in enumerate(self.gen):
			if gen[GEN_STATUS] == 0:
				continue
			bus = self._bus_index[int(gen[GEN_BUS])]
			rows.extend((bus, n + bus))
			columns.extend((2 * n + i, 2 * n + self.n_gen + i))

		row = 2 * n
		for k, branch in enumerate(self.branch):
			f, t, _, _, _, _ = self._branch_y[k]
			if branch[BR_STATUS] == 0:
				continue
			if branch[ANGMIN] > -360:
				rows.extend((row, row))
				columns.extend((f, t))
				row += 1
			if branch[ANGMAX] < 360:
				rows.extend((row, row))
				columns.extend((f, t))
				row += 1
			if branch[RATE_A] > 0:
				for a, b in ((f, t), (t, f)):
					rows.extend((row,) * 4)
					columns.extend((a, b, n + a, n + b))
					row += 1
		coordinates = np.unique(np.column_stack((rows, columns)), axis=0)
		return coordinates[:, 0], coordinates[:, 1]

	def solve_ipopt(self, options=None):
		"""Solve this AC-OPF using cyipopt and return (x, IPOPT info)."""
		try:
			import cyipopt
		except ImportError as error:
			raise ImportError(
				"IPOPT solving requires cyipopt and a system IPOPT installation"
			) from error

		class Callbacks:
			def __init__(callback_self):
				callback_self.rows, callback_self.columns = self.ipopt_jacobian_structure()

			def objective(callback_self, x):
				return self.objective(x)

			def gradient(callback_self, x):
				return self.objective_gradient(x)

			def constraints(callback_self, x):
				return self.ipopt_constraints(x)

			def jacobianstructure(callback_self):
				return callback_self.rows, callback_self.columns

			def jacobian(callback_self, x):
				matrix = self.ipopt_constraint_jacobian(x)
				return np.asarray(
					matrix[callback_self.rows, callback_self.columns]
				).reshape(-1)

		lower_bounds = np.array([
			-np.inf if lower is None else lower for lower, _ in self.bounds
		])
		upper_bounds = np.array([
			np.inf if upper is None else upper for _, upper in self.bounds
		])
		n_equalities = 2 * self.n_bus
		n_constraints = self.ipopt_constraints(self.x0).size
		constraint_lower = np.r_[np.zeros(n_equalities), np.full(
			n_constraints - n_equalities, -np.inf
		)]
		constraint_upper = np.zeros(n_constraints)
		nlp = cyipopt.Problem(
			n=self.dimension,
			m=n_constraints,
			problem_obj=Callbacks(),
			lb=lower_bounds,
			ub=upper_bounds,
			cl=constraint_lower,
			cu=constraint_upper,
		)
		ipopt_options = {
			"print_level": 0,
			"max_iter": 2000,
			"tol": 1e-8,
			"bound_relax_factor": 0.0,
			"hessian_approximation": "limited-memory",
		}
		if options:
			ipopt_options.update(options)
		for name, value in ipopt_options.items():
			nlp.add_option(name, value)
		return nlp.solve(self.x0)

	def salm_inequality_jacobian(self, x):
		"""Sparse Jacobian of salm_inequalities(), with g(x) <= 0."""
		return -self.inequality_jacobian(x)

	def _salm_equality_jacobian_product(self, x, vec):
		return np.asarray(
			self.salm_equality_jacobian(x).T @ np.asarray(vec)
		).reshape(-1)

	def _salm_inequality_jacobian_product(self, x, vec):
		return np.asarray(
			self.salm_inequality_jacobian(x).T @ np.asarray(vec)
		).reshape(-1)


def load_problem(folder, case_name=None):
	"""Load a case by path, or by directory plus case filename/stem.

	Example: ``problem = load_problem("pglib-opf-master", "pglib_opf_case14_ieee")``.
	For SALM, pass ``objective``/``objective_gradient``,
	``salm_equalities``/``jac_h`` and
	``salm_inequalities``/``jac_g``. ``inequalities`` itself returns positive
	feasibility margins, whereas SALM expects nonpositive constraints.
	"""
	path = Path(folder).expanduser()
	if case_name is not None:
		candidate = Path(case_name).expanduser()
		if candidate.is_file():
			path = candidate
		else:
			if candidate.suffix.lower() != ".m":
				candidate = candidate.with_suffix(".m")
			path = path / candidate
	elif path.is_dir():
		raise ValueError("case_name is required when folder is a directory")
	if not path.is_file():
		raise FileNotFoundError(f"MATPOWER case not found: {path}")
	return ACOPFProblem(_read_matpower(path), name=path.stem)
