import torch
import torch.nn as nn
import numpy as np

# Torch Implementation of algorithm listed in 
# 
#  @article{KLINE2019128,
#  title = "Properties of the d-dimensional earth mover’s problem",
#  journal = "Discrete Applied Mathematics",
#  volume = "265",
#  pages = "128 - 141",
#  year = "2019",
#  issn = "0166-218X",
#  doi = "https://doi.org/10.1016/j.dam.2019.02.042",
#  url = "http://www.sciencedirect.com/science/article/pii/S0166218X19301441",
#  author = "Jeffery Kline",
#  keywords = "Submodularity, Monge Property, Linear Programming, Greedy Algorithm, Transportation Problem, Convex Polytopes"}


def OBJ(i):
	return max(i) - min(i)
	# return 0 if max(i) == min(i) else 1

class dEMD(nn.Module):
	def __init__(self, cost=OBJ, computeDual=False, verbose=False):
		super().__init__()

		self.cost = cost
		self.verbose = verbose
		self.computeDual = computeDual

	def forward(self, x):
		d, n = x.shape

		# sum_aa = x.sum(axis=1)
		# assert abs(max(sum_aa)-min(sum_aa)) < 1e-10

		AA = x.clone()

		xx = {}
		if self.computeDual:
			dual = torch.zeros(d,n).double()
		idx = [0,]*d
		obj = 0

		if self.verbose:
			print('i minval oldidx\t\tobj\t\tvals')

		while all([i < n for i in idx]):

			vals = [AA[i,j] for i,j in zip(range(d), idx)]
			minval = min(vals).clone()
			ind = vals.index(minval)
			xx[tuple(idx)] = minval
			obj += (OBJ(idx)) * minval
			for i,j in zip(range(d), idx): AA[i,j] -= minval
			oldidx = np.copy(idx)
			idx[ind] += 1
			if self.computeDual:
				if idx[ind]<n:
					dual[ind,idx[ind]] += self.cost(idx) - self.cost(oldidx) + dual[ind,idx[ind]-1]
			if self.verbose:
				print(ind, minval.item(), oldidx, obj.item(), '\t', vals)

		if self.computeDual:
			for _, i in enumerate(idx):
				try: dual[_][i:] = dual[_][i]
				except: pass

			dualobj =  sum([_.dot(_d) for _, _d in zip(x, dual)])

			return obj, dualobj
		else:
			return obj

