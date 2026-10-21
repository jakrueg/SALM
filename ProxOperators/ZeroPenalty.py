from pyproximal import ProxOperator
import numpy as np

class ZeroPenalty(ProxOperator):
	"""Zero-valued nonsmooth term with the identity proximal operator."""

	def __call__(self, x):
		return 0.0

	def prox(self, x, tau):
		return np.asarray(x, dtype=float).copy()