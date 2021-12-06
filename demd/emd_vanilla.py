# emd_vanilla.py

import numpy as np
from scipy.optimize import approx_fprime

from .emd import greedy_primal_dual

def matricize(x, d, n):
	return np.reshape(x, (d, n))

def listify(x):
	tmp = []
	for i in range(x.shape[0]):
		tmp.append(x[i,:])

	return tmp

def approxGrad(f, x, d, n):
	grad = approx_fprime(x, f, 1e-8, d, n)
	return grad

def demd_func(x, d, n, return_dual_vars=False):
	x = matricize(x, d, n)
	x = listify(x)
	log = greedy_primal_dual(x)

	if return_dual_vars:
		return log['primal objective'], matricize(log['dual'], d, n), log['dual objective']
	else:
		return log['primal objective']
	