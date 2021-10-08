import torch
import numpy as np
import ot
from torch.autograd import grad as agrad
import time

from diff_test import takeStep

def OBJ(i):
    return max(i) - min(i)
    # return 0 if max(i) == min(i) else 1

def torch_greedy_primal_dual(aa, verbose=False):
    # aa is a Torch matrix with d rows, n columns

    d, n = aa.shape

    sum_aa = aa.sum(axis=1)
    # assert abs(max(sum_aa)-min(sum_aa)) < 1e-10

    AA = aa.clone()

    xx = {}
    dual = torch.zeros(d,n)
    idx = [0,]*d
    obj = 0

    if verbose:
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
        if idx[ind]<n:
            dual[ind,idx[ind]] += OBJ(idx) - OBJ(oldidx) + dual[ind,idx[ind]-1]
        if verbose:
            print(ind, minval.item(), oldidx, obj.item(), '\t', vals)

    # the above terminates when any entry in idx equals the corresponding value in dims
    # this leaves other dimensions incomplete; the remaining terms of the dual solution 
    # must be filled-in
    for _, i in enumerate(idx):
        try: dual[_,i:] = dual[_,i]
        except: pass

    dualobj = sum([aa[i,:].dot(dual[i,:]) for i in range(d)])
    
    return {'x': xx, 'primal objective': obj,
            'dual': dual, 'dual objective': dualobj}


def torch_demd_func(x):
    log = torch_greedy_primal_dual(x)
    return log['primal objective']


def minimize(func, x_0, niters=100, lr=0.1, verbose=False):

    x = x_0

    if verbose:
        tic = time.time()
        funcval = func(x)
        print(time.time()-tic, ' seconds for forward.')

        tic = time.time()
        grad, = agrad(outputs=funcval, inputs=x, allow_unused=True)
        print(time.time()-tic, ' seconds for gradient.')
    else:
        funcval = func(x)
        grad, = agrad(outputs=funcval, inputs=x, allow_unused=True)

    gn = torch.linalg.norm(grad)

    print(f'Inital:\t\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    for i in range(niters):

        nx = x / (torch.sum(x, 1).unsqueeze(-1))
        funcval = func(nx)
        
        grad, = agrad(outputs=funcval, inputs=x, allow_unused=True)
        x = takeStep(x, grad, lr)

        gn = np.linalg.norm(grad)
        #funcval = func(x)
        #grad, = agrad(outputs=funcval, inputs=x, allow_unused=True)
        
        if i % 10 == 0:
            print(f'Iter {i:2.0f}:\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    print(x)
    return



if __name__ == "__main__":
    
    np.random.seed(0)

    print('*'*10)
    print('*** 2 Fixed Dists with 6 Bins ***')
    #######
    n = 5  # nb bins
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')

    a1 = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
    a2 = np.array([0.2, 0.1, 0.2, 0.3, 0.2])
    a3 = np.array([0.1, 0.1, 0.5, 0.2, 0.1])
    data = [a1, a2, a3]
    d = len(data)
    print(data)

    # data = np.array(data)
    # data = vectorize(data, vecsize)

    ta1 = torch.Tensor(a1)
    ta2 = torch.Tensor(a2)
    ta3 = torch.Tensor(a3)
    torch_data = [ta1, ta2, ta3]
    torch_data = torch.stack(torch_data).clone().requires_grad_(requires_grad=True)

    # data = data.clone().requires_grad_(requires_grad=True)

    # model = dEMD(epsilon=epsilon)
    func = torch_demd_func
    # func = dEMD(torch_data)
    minimize(func, torch_data, niters=500, lr=0.001, verbose=True)

   
