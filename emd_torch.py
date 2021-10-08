import torch
import torch.nn as nn
import numpy as np

# Torch Implementation of algorithm listed in 
# 
#  @article{KLINE2019128,
#  title = "Properties of the d-dimensional earth mover’s problem",
#  journal = "Discrete Applied Mathematics",
#  volume = "265",
#  pages = "128 - 141",
#  year = "2019",
#  issn = "0166-218X",
#  doi = "https://doi.org/10.1016/j.dam.2019.02.042",
#  url = "http://www.sciencedirect.com/science/article/pii/S0166218X19301441",
#  author = "Jeffery Kline",
#  keywords = "Submodularity, Monge Property, Linear Programming, Greedy Algorithm, Transportation Problem, Convex Polytopes"}

def OBJ(i):
    return max(i) - min(i)
    # return 0 if max(i) == min(i) else 1

class dEMD(nn.Module):
    def __init__(self, cost=OBJ, verbose=False):
        super().__init__()

        self.cost = cost
        self.verbose = verbose

    def forward(self, x):
        d, n = x.shape

        # sum_aa = x.sum(axis=1)
        # assert abs(max(sum_aa)-min(sum_aa)) < 1e-10

        AA = x.clone()

        xx = {}
        dual = torch.zeros(d,n)
        idx = [0,]*d
        obj = 0

        if self.verbose:
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
                dual[ind,idx[ind]] += self.cost(idx) - self.cost(oldidx) + dual[ind,idx[ind]-1]
            if self.verbose:
                print(ind, minval.item(), oldidx, obj.item(), '\t', vals)

        return obj


class FairEMD(nn.Module):
    def __init__(self, cost=OBJ, discretization=10, verbose=False):
        super().__init__()

        self.cost = cost
        self.verbose = verbose


    def forward(output, group_labels):

        # first organize output into distributions.

        # discretize to bins

        gro

        fairObj = self.fairMeasure(grouped_dists)
        return fairObj


if __name__ == '__main__':

    from emd import greedy_primal_dual
    import numpy as np

    np.random.seed(0)

    print('*'*10)
    print('*** 2 Fixed Dists with 6 Bins ***')
    #######
    n = 5  # nb bins

    a1 = np.array([0.5, 0.2, 0.1, 0.1, 0.1], dtype=np.float32)
    a2 = np.array([0.2, 0.1, 0.2, 0.3, 0.2], dtype=np.float32)
    a3 = np.array([0.1, 0.1, 0.5, 0.2, 0.1], dtype=np.float32)
    data = [a1, a2, a3]
    d = len(data)
    print(data)

    # numpy version
    print('Numpy Version:')
    tmp = greedy_primal_dual(data, verbose=True)
    print(tmp)

    ta1 = torch.Tensor(a1)
    ta2 = torch.Tensor(a2)
    ta3 = torch.Tensor(a3)
    torch_data = [ta1, ta2, ta3]

    # torch version
    print('Torch Version:')
    tmp = torch_greedy_primal_dual(torch.stack(torch_data), verbose=True)
    print(tmp)

