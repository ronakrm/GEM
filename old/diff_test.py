import numpy as np
from scipy.optimize import approx_fprime
import ot

from emd import greedy_primal_dual

n = 5
d = 3
vecsize = n*d

def vectorize(x, vecsize):
    return np.reshape(x, vecsize)

def matricize(x, n, d):
    return np.reshape(x, (n, d))

def listify(x):
    tmp = []
    for i in range(x.shape[0]):
        tmp.append(x[i,:])

    return tmp

def demd_func(x):
    x = matricize(x, d, n)
    x = listify(x)
    log = greedy_primal_dual(x)
    return log['primal objective']

def approxGrad(f, x):
    grad = approx_fprime(x, f, epsilon=1e-8)
    return grad


def takeStep(x, grad, lr=0.1):
    return x - lr*grad

def renormalize(x):
    x = matricize(x, d, n)
    for i in range(x.shape[0]):
        x[i,:] = x[i,:]/np.sum(x[i,:])
    return vectorize(x, vecsize)

def minimize(f, x_0, niters=100, lr=0.1):

    x = x_0
    funcval = f(x)
    grad = approxGrad(f, x)
    gn = np.linalg.norm(grad)

    # print(f'Inital:\t\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    for i in range(niters):

        x = renormalize(x)
        funcval = f(x)
        
        grad = approxGrad(f, x)
        x = takeStep(x, grad, lr)

        print(grad)
        gn = np.linalg.norm(grad)
                
        if i % 1 == 0:
            print(f'Iter {i:2.0f}:\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    print('d1:', x[:5])
    print('d2:', x[5:10])
    print('d2:', x[10:])
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

    vecsize = n*d

    data = np.array(data)
    data = vectorize(data, vecsize)

    minimize(demd_func, data, niters=10, lr=0.0001)

   
