# grad_test.py
import argparse

import pandas as pd
import time

import numpy as np
import ot

from scipy.optimize import approx_fprime
from emd import greedy_primal_dual
from demdLayer import DEMDLayer
from emd_torch import dEMD

def genNumpyData(n, d):

	data = []

	# Gaussianlike distributions
	data = []
	for i in range(d):
		m = 100*np.random.rand(1)
		a = ot.datasets.make_1D_gauss(n, m=m, s=5)
		print(a)
		import pdb; pdb.set_trace()
		data.append(a)

	return data

def test(n, d, seed, gradType, outfile):

	tmp = {}
	tmp['n'] = [n]
	tmp['d'] = [d]
	tmp['seed'] = [random_seed]
	tmp['gradType'] = [epsilon]

	np_data = genNumpyData(n, d)

	# npdata = np.array(data)
	# d,n = npdata.shape
	# x = np.reshape(npdata, d*n)

	# funcval = demd_func(x, d, n)
	# grad = approxGrad(demd_func, x, d, n)
	# grad = matricize(grad, d, n)
	# grad = listify(grad)
	
	# print('scipy approx grad:')
	# print(np.round(grad))

	# print('dual variables:')
	# print(greedy_primal_dual(data)['dual'])

	# print('obj:')
	# print(greedy_primal_dual(data)['primal objective'])

	tmp['forward_time'] = ['a']
	tmp['backward_time'] = ['b']

	df = pd.DataFrame(tmp)
	if os.path.isfile(outfile):
		df.to_csv(outfile, mode='a', header=False, index=False)
	else:
		df.to_csv(outfile, mode='a', header=True, index=False)

	return


def main(args):
	test(args.n, args.d, args.random_seed, args.gradType)


if __name__ == "__main__":

	arg_parser = argparse.ArgumentParser(description='Speed Test')

	arg_parser.add_argument('--random_seed', type=int, default=0)
	arg_parser.add_argument('--n', type=int, default=5)
	arg_parser.add_argument('--d', type=int, default=3)

	arg_parser.add_argument('--gradType', type=str, default='scipy',
						choices=['scipy', 'npdual', 'torchdual', 'autograd'])

	arg_parser.add_argument('--outfile', type=str, default='speed_test_results.csv')

	args = arg_parser.parse_args()
	main(args)