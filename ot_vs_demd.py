import numpy as np

import ot

from emd import greedy_primal_dual

from emd_cvxopt import cvxprimal


def breg_1d_bary(data, M, n, d):

	A = data


	alpha = 1.0#/d  # 0<=alpha<=1
	weights = np.array(d*[alpha]) 
 
	# l2bary
	bary_l2 = A.dot(weights)

	# wasserstein
	reg = 1e-3
	#_, bary_wass_log = ot.bregman.barycenter(A, M, reg, weights, verbose=False, log=True)
	_, bary_wass2_log = ot.lp.barycenter(A, M, weights, solver='interior-point', verbose=False, log=True)

	#return bary_wass_log['err'][-1]
	return bary_wass2_log['fun']


def demd(data):
	return greedy_primal_dual(data)['primal objective']


####### Two known dists
#######
d = 2 # n samples/dimensions
n = 4  # nb bins
a1 = np.array([0.25, 0.25, 0.25, 0.25])
a2 = np.array([0.05, 0.25, 0.25, 0.45])
print(a1)
print(a2)
M = np.array([[0,1,2,3.0],[1,0,1,2],[2,1,0,1],[3,2,1,0]])
ot.tic()
print('breg POT 1d bary obj\t: ', breg_1d_bary(np.vstack((a1, a2)).T, M, n, d))
ot.toc()
ot.tic()
print('demd obj\t\t: ', demd([a1, a2]))
ot.toc()
#######


####### Two random dists
#######
d = 2 # n samples/dimensions
n = 4  # nb bins
# Gaussian distributions
a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)
print(a1)
print(a2)
M = np.array([[0,1,2,3.0],[1,0,1,2],[2,1,0,1],[3,2,1,0]])
ot.tic()
print('breg POT 1d bary obj\t: ', breg_1d_bary(np.vstack((a1, a2)).T, M, n, d))
ot.toc()
ot.tic()
print('demd obj\t\t: ', demd([a1, a2]))
ot.toc()
#######

####### Four random dists, 10 bins
#######
d = 4
n = 10  # nb bins

# Gaussian distributions
a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)
a3 = ot.datasets.make_1D_gauss(n, m=40, s=3)
a4 = ot.datasets.make_1D_gauss(n, m=20, s=25)
x = np.arange(n, dtype=np.float64).reshape((n, 1))
M = ot.utils.dist(x, metric='minkowski')
print(M)
ot.tic()
print('breg POT 1d bary obj\t: ', breg_1d_bary(np.vstack((a1, a2, a3, a4)).T, M, n, d))
ot.toc()
ot.tic()
print('demd obj\t\t: ', demd([a1, a2, a3, a4]))
ot.toc()
ot.tic()
cvx_aa = cvxprimal([a1, a2, a3, a4])
print('  cvxopt primal obj      : %6.4f' % cvx_aa['primal objective'])
ot.toc()
#######
