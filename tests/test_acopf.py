import unittest
from pathlib import Path

import numpy as np

from acopf import load_problem


class ACOPFDerivativeTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		root = Path(__file__).resolve().parents[1]
		cls.problem = load_problem(root / "pglib-opf-master", "pglib_opf_case14_ieee")
		rng = np.random.default_rng(12)
		cls.x = cls.problem.x0 + rng.normal(scale=0.015, size=cls.problem.dimension)
		cls.step = 1e-6

	def finite_difference_jacobian(self, function):
		n = self.problem.dimension
		identity = np.eye(n)
		return np.column_stack([
			(function(self.x + self.step * identity[i])
			 - function(self.x - self.step * identity[i])) / (2 * self.step)
			for i in range(n)
		])

	def test_objective_gradient(self):
		problem = self.problem
		numeric = np.array([
			(problem.objective(self.x + self.step * np.eye(problem.dimension)[i])
			 - problem.objective(self.x - self.step * np.eye(problem.dimension)[i]))
			/ (2 * self.step)
			for i in range(problem.dimension)
		])
		np.testing.assert_allclose(problem.objective_gradient(self.x), numeric, atol=1e-5)

	def test_equality_jacobians(self):
		problem = self.problem
		numeric = self.finite_difference_jacobian(problem.salm_equalities)
		np.testing.assert_allclose(
			problem.salm_equality_jacobian(self.x).toarray(),
			numeric,
			rtol=1e-7,
			atol=1e-4,
		)

	def test_inequality_jacobians_and_salm_sign(self):
		problem = self.problem
		numeric = self.finite_difference_jacobian(problem.inequalities)
		jacobian = problem.inequality_jacobian(self.x).toarray()
		np.testing.assert_allclose(jacobian, numeric, rtol=1e-7, atol=1e-4)
		np.testing.assert_allclose(
			problem.salm_inequalities(self.x), -problem.inequalities(self.x)
		)
		np.testing.assert_allclose(
			problem.salm_inequality_jacobian(self.x).toarray(), -jacobian
		)

	def test_matrix_free_jacobian_products(self):
		problem = self.problem
		equality_vector = np.ones(problem.salm_equalities(self.x).size)
		inequality_vector = np.ones(problem.salm_inequalities(self.x).size)
		np.testing.assert_allclose(
			problem.jac_h.jacobian_vector_product(self.x, equality_vector),
			problem.salm_equality_jacobian(self.x).T @ equality_vector,
		)
		np.testing.assert_allclose(
			problem.jac_g.jacobian_vector_product(self.x, inequality_vector),
			problem.salm_inequality_jacobian(self.x).T @ inequality_vector,
		)

	def test_ipopt_constraint_jacobian_structure(self):
		problem = self.problem
		rows, columns = problem.ipopt_jacobian_structure()
		jacobian = problem.ipopt_constraint_jacobian(self.x)
		values = np.asarray(jacobian[rows, columns]).reshape(-1)
		reconstructed = np.zeros(jacobian.shape)
		reconstructed[rows, columns] = values
		np.testing.assert_allclose(reconstructed, jacobian.toarray())

		numeric = self.finite_difference_jacobian(problem.ipopt_constraints)
		np.testing.assert_allclose(jacobian.toarray(), numeric, rtol=1e-7, atol=1e-4)


if __name__ == "__main__":
	unittest.main()
