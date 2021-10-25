import torch
import numpy as np

from demdLayer import DEMDLayer


if __name__ == "__main__":

	print('*'*10)    
	np.random.seed(0)


	### Data

	n = 100  # number of samples
	d = 5 # number of groups

	group_labels = np.random.randint(0, d, size=(n,))

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

	groups = np.unique(group_labels)
	for i in range(d):
		idxs = group_labels==groups[i]
		print(tacts[idxs])
	
	### Model/Layer
	myl = DEMDLayer()

	opt = torch.optim.SGD([tacts], lr=0.1)

	### Forward
	print(myl(tacts, group_labels))

	### Backward/Opt
	n_epochs = 100
	for t in range(n_epochs):

		res = myl(tacts, group_labels)
		print(res)

		opt.zero_grad()
		res.backward()
		opt.step()

	groups = np.unique(group_labels)
	for i in range(d):
		idxs = group_labels==groups[i]
		print(tacts[idxs])