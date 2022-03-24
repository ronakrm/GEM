import torch
import numpy as np

from demd import DEMDLayer
from src.measures.barywas import WassersteinBarycenter


if __name__ == "__main__":

	print('*'*10)    
	np.random.seed(0)


	### Parameters
	n = 100  # number of samples
	d = 2 # number of groups
	nbins = 10
	lr = 0.01
	n_epochs = 10000

	### Data
	group_labels = torch.from_numpy(np.random.randint(0, d, size=(n,)))

	acts = []
	for i in range(n):
		if group_labels[i] == 0:
			val = -1+np.random.rand(1)
		if group_labels[i] == 1:
			val = +1+np.random.rand(1)
		else:
			val = (np.random.rand(1)-0.5)*2
		
		acts.append(val)

	tacts = torch.Tensor(np.array(acts)).requires_grad_(requires_grad=True)

	### Model/Layer
	myl = DEMDLayer(discretization=nbins)
	# myl = WassersteinBarycenter(discretization=nbins)

	groups = np.unique(group_labels)
	for i in range(d):
		idxs = group_labels==groups[i]
		print(myl.genHists(tacts[idxs],nbins=nbins))


	opt = torch.optim.SGD([tacts], lr=lr)

	### Forward
	# print(myl(tacts, group_labels))

	### Backward/Opt
	for t in range(n_epochs):

		res = myl(tacts, group_labels)

		if t % 100 == 0:
			print(res)

		opt.zero_grad()
		res.backward()
		opt.step()

	groups = np.unique(group_labels)
	for i in range(d):
		idxs = group_labels==groups[i]
		print(myl.genHists(tacts[idxs],nbins=10))
