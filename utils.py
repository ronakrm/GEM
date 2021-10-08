# utils.py
import numpy as np
import ot

def shuffleEMD(data, costs):
	# d is number of points, n bins
	n = costs.shape[1]
	perm = np.random.permutation(n)

	shuf_data = shuf_data[:,perm]
	shuf_costs = costs[np.ix_(perm, perm)]
	return shuf_data, shuf_costs

def getCostTensor(n):
	x = np.arange(n, dtype=np.float64).reshape((n, 1))
	M = ot.utils.dist(x, metric='minkowski')
	return M