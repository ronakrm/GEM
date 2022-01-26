# wasserstein barycenter via POT + Torch Backend
import torch
import torch.nn as nn

import ot

from demd import HistoBin

def OBJ(i):
	return max(i) - min(i)
	# return 0 if max(i) == min(i) else 1

class WassersteinBarycenter(nn.Module):
	def __init__(self, cost=OBJ, discretization=10, verbose=False):
		super().__init__()

		self.cost = cost
		self.verbose = verbose
		self.discretization = discretization
		
		self.cdf = nn.Sigmoid()
		self.Hist = HistoBin(nbins=discretization)

		self.fairMeasure = dEMD()

	def forward(self, acts, group_labels):
		groups = torch.unique(group_labels)
		d = len(groups)
		# first organize output into distributions.
		grouped_dists = []
		for i in range(d):
			idxs = group_labels==groups[i]
			g_dist = self.genHists(acts[idxs], nbins=self.discretization)
			grouped_dists.append(g_dist)

		torch_dists = torch.stack(grouped_dists).requires_grad_(requires_grad=True)

		# convert to numpy, compute barycenter

		# convert bary to torch const

		# sum ot gpu grad distsances to bary

		fairObj = 
		for i in range(d):
			fairObj += ot.emd2(grouped_dists[i], torchBary)

		return fairObj

	def genHists(self, samples, nbins=10):
		# convert to [0,1] via sigmoid
		cdfs = self.cdf(samples)
		dist = self.Hist(cdfs)
		# dist = torch.histc(cdfs, bins=nbins, min=0, max=1)
		return dist/dist.sum()