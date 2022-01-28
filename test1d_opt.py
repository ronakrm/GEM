import argparse
import numpy as np
import os
import pandas as pd

from demd.emd_vanilla import demd_func, minimize
from demd.datagen import getData

def main(args):

	run_dict = vars(args)

	n = args.n  # nb bins
	d = args.d

	vecsize = n*d

	# data, M = getData(n, d, 'uniform')
	import pdb; pdb.set_trace()
	data, M = getData(n, d, 'skewedGauss')

	import pdb; pdb.set_trace()
	from demd.emd_utils import lp_1d_bary, sink_1d_bary
	obj, bary = lp_1d_bary(data, M, n, d)
	obj, bary = sink_1d_bary(data, M, n, d)


	x = minimize(demd_func, data, d, n, vecsize,
	                 niters=args.iters, lr=args.learning_rate)

	for i, y in enumerate(x):
		run_dict['group'] = i
		for j, v in enumerate(y):
			run_dict['bin'] = j
			run_dict['val'] = v

			df = pd.DataFrame.from_records([run_dict])	
			if os.path.isfile(args.outfile):
				df.to_csv(args.outfile, mode='a', header=False, index=False)
			else:
				df.to_csv(args.outfile, mode='a', header=True, index=False)


if __name__ == "__main__":
	arg_parser = argparse.ArgumentParser(description='Distance Compute')
	arg_parser.add_argument('--seed', type=int, default=0)
	arg_parser.add_argument('-n', type=int, default=50, help='Number of bins')
	arg_parser.add_argument('-d', type=int, default=7, help='Number of dists')
	arg_parser.add_argument('--iters', type=int, default=0)
	arg_parser.add_argument('--learning_rate', type=float, default=1e-6)
	arg_parser.add_argument('--outfile', type=str, default='results/1d_hist_results.csv', help='results file to print to')
	args = arg_parser.parse_args()

	np.random.seed(args.seed)
	main(args)


############# OLD

	# print(x)
	# print(sum(x[0]))
	# print(sum(x[1]))
	# print(sum(x[2]))
	# print(time)

	# import pdb; pdb.set_trace()


	# device = 'cpu'
	# imgs = torch.tensor(imgs_np, dtype=torch.float64, device=device,
	#                     requires_grad=False)
	# # dists = create_distribution_2d(imgs_np)
	# imgs = imgs + 1e-10
	# imgs /= imgs.sum((1, 2))[:, None, None]
	# epsilon = 0.002

	# grid = torch.arange(width).type(torch.float64)
	# grid /= width
	# M = (grid[:, None] - grid[None, :]) ** 2
	# M_large = M[:, None, :, None] + M[None, :, None, :]
	# M_large = M_large.reshape(n_features, n_features)
	# M_large = M_large.to(device)

	# K = torch.exp(- M / epsilon)
	# K = K.to(device)
	# # 
	# # print("Doing IBP ...")
	# # time_ibp = time.time()
	# bar_ibp, log = barycenter(imgs, K, reference="uniform", return_log=True)
	# # time_ibp = time.time() - time_ibp
	# print('IBP', log['a'])
