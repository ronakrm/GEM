import numpy as np

import ot

from emd_utils import compare_all

def known2d_simple():

    ####### Two known dists
    print('\n')
    print('*'*10)
    print('*** 2 Fixed Dists with 4 Bins ***')
    #######
    d = 3 # n samples/dimensions
    n = 2  # nb bins
    a1 = np.array([0.2, 0.8])
    a2 = np.array([0.4, 0.6])
    a3 = np.array([0.4, 0.6])
    #a1 = np.array([0.2, 0.8])
    #a2 = np.array([0.8, 0.2])
    print(a1)
    print(a2)
    print(a3)
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')
    #M = np.array([[0,1,1,1.0],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
    compare_all([a1, a2, a3], M, n, d)
    print('*'*10)
    print('\n')
    #######

    return

def known2d():

    ####### Two known dists
    print('\n')
    print('*'*10)
    print('*** 2 Fixed Dists with 2 Bins ***')
    #######
    d = 2 # n samples/dimensions
    n = 4  # nb bins
    a1 = np.array([0.25, 0.25, 0.25, 0.25])
    a2 = np.array([0.05, 0.25, 0.25, 0.45])
    print(a1)
    print(a2)
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')
    #M = np.array([[0,1,1,1.0],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
    compare_all([a1, a2], M, n, d)
    print('*'*10)
    print('\n')
    #######

    return

def increasing_bins():

    ns = [5, 10, 20, 50, 100]
    for n in ns:
        random2d(n=n)

    return

def random2d(n=4):

    ####### Two random dists
    print('\n')
    print('*'*10)
    print('*** 2 Random Dists with 4 Bins ***')
    #######
    d = 2 # n samples/dimensions
    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)
    print(a1)
    print(a2)
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')
    print(M)
    compare_all([a1, a2], M, n, d)
    print('*'*10)
    print('\n')
    #######

    return

def random4d():

    ####### Four random dists, 10 bins
    print('\n')
    print('*'*10)
    print('*** 4 Random Dists with 10 Bins ***')
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
    compare_all([a1, a2, a3, a4], M, n, d)
    print('*'*10)
    print('\n')
    #######

    return

if __name__ == "__main__":

    np.random.seed(0)

    known2d_simple()
    #known2d()
    #random2d()
    #random4d()


    #increasing_bins()



