import numpy as np

import ot

from emd import greedy_primal_dual

from emd_cvxopt import cvxprimal

def get_2d_emd_dist(x, y, M):
    gamma, log = ot.lp.emd(x, y, M, log=True)

    return log['cost']

def sum_2d_emd_dists(As, bary, M):

    tmp = 0.0
    for x in As:
        tmp += get_2d_emd_dist(x, bary, M)

    return tmp
    
def sink_1d_bary(data, M):

	A = data

	reg = 1e-2

	bary, bary_log = ot.barycenter(A, M, reg, weights=None, verbose=False, log=True)
	return bary, bary_log['err'][-1]

def lp_1d_bary(data, M, n, d):

	A = data

	alpha = 1.0#/d  # 0<=alpha<=1
	weights = np.array(d*[alpha]) 

	bary, bary_log = ot.lp.barycenter(A, M, weights, solver='interior-point', verbose=False, log=True)

	return bary, bary_log['fun']


def demd(data):
    log = greedy_primal_dual(data)
    return log['x'], log['primal objective']

def compare_all(data, M, n, d):

    print('\tSinkhorn Iterations:')
    ot.tic()
    sink_bary, sink_obj = sink_1d_bary(np.vstack(data).T, M)
    sink_time = ot.toc('')
    #print('\t Obj\t: ', sink_obj)
    #print('SumBaryDist\t: ', sum_2d_emd_dists(data, sink_bary/np.sum(sink_bary), M))
    print('\t Obj\t: ', sum_2d_emd_dists(data, sink_bary/np.sum(sink_bary), M))
    print('\t Bary\t: ', sink_bary)
    print('\t Time\t: ', sink_time)

    print('')
    print('\tIP LP Iterations:')
    ot.tic()
    ip_bary, ip_obj = lp_1d_bary(np.vstack(data).T, M, n, d)
    ip_time = ot.toc('')
    print('\t Obj\t: ', ip_obj)
    print('\t Bary\t: ', ip_bary)
    print('\t Time\t: ', ip_time)

    print('')
    print('\tD-EMD Algorithm:')
    ot.tic()
    demd_bary, demd_obj = demd(data)
    demd_time = ot.toc('')
    print('\t Obj\t: ', demd_obj)
    print('\t Bary\t: ', demd_bary)
    print('\t Time\t: ', demd_time)

    # print('')
    # print('\tFull CVXOPT:')
    # ot.tic()
    # cvx_obj = cvxprimal(data)['primal objective']
    # cvx_time = ot.toc('')
    # print('\t Obj\t: ', cvx_obj)
    # print('\t Time\t: ', cvx_time)


