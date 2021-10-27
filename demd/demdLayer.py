import torch
import torch.nn as nn
import numpy as np

from emd_torch import dEMD

def OBJ(i):
	return max(i) - min(i)
	# return 0 if max(i) == min(i) else 1

class DEMDLayer(nn.Module):
	def __init__(self, cost=OBJ, discretization=10, verbose=False):
		super().__init__()

		self.cost = cost
		self.verbose = verbose
		self.discretization = discretization
		
		self.cdf = nn.Sigmoid()
		self.Hist = HistoBin(nbins=discretization)

		self.fairMeasure = dEMD()

	def forward(self, acts, group_labels):
		groups = np.unique(group_labels)
		d = len(groups)
		# first organize output into distributions.
		grouped_dists = []
		for i in range(d):
			idxs = group_labels==groups[i]
			g_dist = self.genHists(acts[idxs], nbins=self.discretization)
			grouped_dists.append(g_dist)

		torch_dists = torch.stack(grouped_dists).requires_grad_(requires_grad=True)
		fairObj = self.fairMeasure(torch_dists)
		return fairObj


	def genHists(self, samples, nbins=10):
		# convert to [0,1] via sigmoid
		cdfs = self.cdf(samples)
		dist = self.Hist(cdfs)
		# dist = torch.histc(cdfs, bins=nbins, min=0, max=1)
		return dist/dist.sum()

class HistoBin(nn.Module):
	def __init__(self, nbins, norm=True):
		super(HistoBin, self).__init__()
		self.locs = np.arange(0,1,1.0/nbins)
		self.r = 1.0/nbins
		self.norm = norm
	
	def forward(self, x):
		
		counts = []
		
		for loc in self.locs:
			dist = torch.abs(x - loc)
			#print dist
			ct = torch.relu(self.r - dist).sum() 
			counts.append(ct)
		
		# out = torch.stack(counts, 1)
		out = torch.stack(counts)
		
		if self.norm:
			summ = out.sum() + .000001
			out = out / summ
			# return (out.transpose(1,0) / summ).transpose(1,0)
		return out
