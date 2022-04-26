import torch
import numpy as np
import time
import ot
import argparse
import pandas as pd
import os

from demd import DEMDLayer
from src.measures.barywas import WassersteinBarycenter


def test(params):

	print('*'*10)    
	np.random.seed(params.seed)


	### Parameters
	run_dict = vars(args)

	### Data
	group_labels = torch.from_numpy(np.random.randint(0, params.d, size=(params.n,)))

	acts = []
	for i in range(params.n):
		if group_labels[i] == 0:
			val = -1+np.random.rand(1)
		if group_labels[i] == 1:
			val = +1+np.random.rand(1)
		else:
			val = (np.random.rand(1)-0.5)*2
		
		acts.append(val)

	tacts = torch.Tensor(np.array(acts)).requires_grad_(requires_grad=True)

	### Model/Layer
	if params.distType == 'demd':
		myl = DEMDLayer(discretization=params.nbins)
	elif params.distType == 'pairwass':
		myl = WassersteinBarycenter(discretization=nbins)

	groups = np.unique(group_labels)
	for i in range(params.d):
		idxs = group_labels==groups[i]
		print(myl.genHists(tacts[idxs],nbins=params.nbins))


	opt = torch.optim.SGD([tacts], lr=params.learning_rate)

	tic = time.time()
	for t in range(params.n_epochs):
		### Model/Layer
		if params.distType == 'demd':
			res = myl(tacts, group_labels)
		elif params.distType == 'pairwass':
			res, bary_est = myl(tacts, group_labels)
		
		if t % 100 == 0:
			print(t, res.item())

		opt.zero_grad()
		res.backward()
		opt.step()

		if params.distType == 'pairwass':
			with torch.no_grad():
				grad = bary_est.grad
				bary_est -= bary_est.grad * params.learning_rate  # step
				bary_est.grad.zero_()
				bary_est.data = ot.utils.proj_simplex(bary_est)  # projection onto the simplex

	toc = time.time()
	groups = np.unique(group_labels)
	for i in range(params.d):
		idxs = group_labels==groups[i]
		print(myl.genHists(tacts[idxs],nbins=params.nbins))
	print(f'Took {toc-tic} seconds.')

	run_dict['loss'] = res.item()
	run_dict['total_time'] = toc-tic
	run_dict['time_per_epoch'] = (toc-tic) / params.n_epochs

	df = pd.DataFrame.from_records([run_dict])
	if os.path.isfile(args.outfile):
		df.to_csv(args.outfile, mode='a', header=False, index=False)
	else:
		df.to_csv(args.outfile, mode='a', header=True, index=False)

if __name__ == "__main__":

	arg_parser = argparse.ArgumentParser(description='Distance Compute')
	arg_parser.add_argument('--seed', type=int, default=0)
	arg_parser.add_argument('-n', type=int, default=10, help='Number of samples (in batch)')
	arg_parser.add_argument('--nbins', type=int, default=10, help='Number of bins')
	arg_parser.add_argument('-d', type=int, default=4, help='Number of dists')
	arg_parser.add_argument('--distType', type=str, default='demd', choices=['demd', 'pairwass'])
	arg_parser.add_argument('--n_epochs', type=int, default=5)
	arg_parser.add_argument('--learning_rate', type=float, default=0.01)
	arg_parser.add_argument('--outfile', type=str, default='results/wassimplresults.csv', help='results file to print to')
	args = arg_parser.parse_args()

	test(args)





