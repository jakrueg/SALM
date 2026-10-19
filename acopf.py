"""Load MATPOWER/PGLib AC-OPF cases as objective and constraint functions.

The decision vector is ``[Va (rad), Vm (p.u.), Pg (MW), Qg (MVAr)]``.
Inequalities follow ``g(x) >= 0``.  The resulting problem provides
``objective``, ``equalities``, ``inequalities``, ``x0`` and ``bounds``.
"""

from pathlib import Path
import re

import numpy as np


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
		if len(self._bus_index) != self.n_bus:
			raise ValueError("MATPOWER bus numbers must be unique")
		if len(self.gencost) < self.n_gen:
			raise ValueError("gencost must contain one row per generator")
		self._build_admittance()
		self.x0, self.bounds = self._make_initial_point_and_bounds()

	@property
	def dimension(self):
		return 2 * self.n_bus + 2 * self.n_gen

	def _build_admittance(self):
		ybus = np.zeros((self.n_bus, self.n_bus), dtype=complex)
		for i, bus in enumerate(self.bus):
			ybus[i, i] += (bus[GS] + 1j * bus[BS]) / self.base_mva
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
			ybus[f, f] += yff
			ybus[f, t] += yft
			ybus[t, f] += ytf
			ybus[t, t] += ytt
			self._branch_y.append((f, t, yff, yft, ytf, ytt))
		self.ybus = ybus

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

	def inequalities(self, x):
		"""Voltage, generation, branch-flow and angle margins (all >= 0)."""
		va, vm, pg, qg = self._unpack(x)
		margins = [vm - self.bus[:, VMIN], self.bus[:, VMAX] - vm]
		for i, gen in enumerate(self.gen):
			if gen[GEN_STATUS] != 0:
				margins.extend((np.array([pg[i] - gen[PMIN]]),
								np.array([gen[PMAX] - pg[i]]),
								np.array([qg[i] - gen[QMIN]]),
								np.array([gen[QMAX] - qg[i]])))
		voltage = vm * np.exp(1j * va)
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
				margins.extend((np.array([row[RATE_A]**2 - abs(sf)**2]),
								np.array([row[RATE_A]**2 - abs(st)**2)))
		return np.concatenate(margins).astype(float)


def load_problem(folder, case_name=None):
	"""Load a case by path, or by directory plus case filename/stem.

	Example: ``problem = load_problem("pglib-opf-master", "pglib_opf_case14_ieee")``.
	Pass ``problem.objective``, ``problem.equalities`` and
	``problem.inequalities`` to SALM; use ``problem.x0`` and ``problem.bounds``
	as the starting point and variable bounds.
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


import_problem = load_problem
