
import torch
import numpy as np
import time

from emd_torch import dEMD

def minimize(func, x_0, niters=100, lr=0.1, verbose=False):

    x = x_0
    
    opt = torch.optim.SGD([x], lr=lr)

    if verbose:
        tic = time.time()
        funcval = func(x)
        print(time.time()-tic, ' seconds for forward.')

        tic = time.time()
        funcval.backward()
        print(time.time()-tic, ' seconds for gradient.')


    for i in range(niters):

        nx = x / (torch.sum(x, 1).unsqueeze(-1))
        funcval = func(nx)

        opt.zero_grad()
        funcval.backward()
        opt.step()

        gn = np.linalg.norm(x.grad)
        
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

    func = dEMD()
    minimize(func, torch_data, niters=500, lr=0.001, verbose=True)

   